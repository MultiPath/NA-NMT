import torch
import numpy as np
import math
import time
import sys

from torch.autograd import Variable
from tqdm import tqdm, trange
from model import Transformer, FastTransformer, SimultaneousTransformer, TINY, positional_encodings_like
from utils import Metrics, Best, computeGLEU, computeBLEU, get_delay, get_delay_online

from ez_train import *

def print_action(traj, traj_mask, losses, quality, delay, reward, index=0):
    action_str = []
    for j in range(traj.size(-2)):
        action_str += ["\n"]
        for i in range(traj.size(-1)):
            if (traj_mask[index, j, i] > 0):
                action_str += ["R" if traj[index, j, i] == 0 else "W"]
        action_str += ['\t(loss) {:.3f}\t(quality) {:.3f}\t(delay) {:.3f}\t(reward) {:.3f}'.format(
                        losses[index, j].data.cpu().numpy()[0], 
                        quality[index, j].data.cpu().numpy()[0], 
                        delay[index, j].data.cpu().numpy()[0],
                        reward[index, j].data.cpu().numpy()[0],
                        ),]
    return "".join(action_str)

def shaping(rewards, dim=0, min_var=1):
    rewards_shaped = rewards - rewards.mean(dim, keepdim=True)
    if rewards_shaped.size(dim) > 1:
        reward_std = rewards_shaped.std(dim, keepdim=True)
    else:
        reward_std = rewards_shaped
    rewards_shaped /= torch.clamp(reward_std, min=min_var)
    return rewards_shaped


def score_model(model, 
                sources, source_masks,
                inputs, input_masks, 
                targets, target_masks,
                traj_source_mask):
    
    encoding = model.encoding(sources, source_masks)
    traj_source_mask = traj_source_mask.transpose(2, 3)
    B, N, Ly, Lx = traj_source_mask.size()
    
    # expand and reshaping everything
    inputs = inputs[:, None, :].expand(B, N, Ly).contiguous().view(B * N, Ly)
    input_masks = input_masks[:, None, :].expand(B, N, Ly).contiguous().view(B * N, Ly)
    targets = targets[:, None, :].expand(B, N, Ly).contiguous().view(B * N, Ly)
    target_masks = target_masks[:, None, :].expand(B, N, Ly).contiguous().view(B * N, Ly)
    traj_source_mask = traj_source_mask.contiguous().view(B * N, Ly, Lx)
    
    new_encoding = []
    for i in range(len(encoding)):
        d = encoding[i].size()[-1]
        new_encoding.append(encoding[i][:, None, :, :].expand(B, N, Lx, d).contiguous().view(B * N, Lx, d))

    # compute the loss
    traj_decoder_out, traj_decoder_probs = model(new_encoding, traj_source_mask, inputs, input_masks, return_probs=True)
    cost, loss = model.batched_cost(targets, target_masks, traj_decoder_probs, batched=True)
    return traj_decoder_out, traj_decoder_probs, cost, loss.view(B, N)


def get_path_loss(probs, path_mask, samples, entropy_weight=0.0):
    losses = -(torch.log(probs + TINY) * (samples - entropy_weight * probs) + 
               torch.log((1 - probs) + TINY) * ((1 - samples) - entropy_weight * (1 - probs)))
    losses = (losses * path_mask).sum(dim=-1).sum(dim=-1) / (path_mask.sum(dim=-1).sum(dim=-1) + TINY)
    return losses

def get_traj_loss(probs, traj_mask, traj, entropy_weight=0.0):
    losses = -(torch.log(probs + TINY) * (traj - entropy_weight * probs) + 
               torch.log((1 - probs) + TINY) * ((1 - traj) - entropy_weight * (1 - probs)))
    losses = (losses * traj_mask).sum(dim=-1) / (traj_mask.sum(dim=-1) + TINY)
    return losses


def step(args, model, sampler=None, actor=None, pre_model=None,
        batch=None, n=10, stochastic=True, rescoring=True, print_out=False):

    if pre_model is None:
        pre_model = model
    
    outputs = dict()

    # pre-model information, prepare data and encoding
    inputs, input_masks, \
    targets, target_masks, \
    sources, source_masks, \
    encoding, batch_size = pre_model.quick_prepare(batch)
    
    encoder_out = encoding[-1]
    decoder_out = pre_model(encoding, source_masks, inputs, input_masks) 
    lx = encoder_out.size(1)
    ly = decoder_out.size(1)
    
    # run the inference-sampler
    probs, masks, samples, traj, traj_mask, path_mask, traj_src_mask = \
        sampler(encoder_out, decoder_out, source_masks, target_masks, n, stochastic, sample=True)
    path_mask = Variable(path_mask)

    outputs['traj_mask'] = traj_mask
    outputs['traj'] = traj
    outputs['batch_size'] = batch_size
   
    # run the scoring function for simultaneous translator
    if rescoring:
        decoder_traj, decoder_probs, traj_cost, traj_loss = score_model(model, sources, source_masks, 
                                                                        inputs, input_masks, targets, 
                                                                        target_masks, traj_src_mask)
        
        # get rewards: (1 - t) * quality + t * delay
        quality = traj_loss
        delay = get_delay(traj_src_mask, masks, type=args.delay_type)
        reward = -((1 - args.delay_weight) * quality + args.delay_weight * delay)

        # reward_shaping on each loss
        quality_shaped = shaping(quality, 1)
        delay_shaped = shaping(delay, 1)
        reward_shaped = -((1 - args.delay_weight) * quality_shaped + args.delay_weight * delay_shaped)

        # get the loss for the sampler
        losses = get_path_loss(probs, path_mask, samples, 0.01)
        loss = (losses * reward_shaped.detach()).mean()

        outputs['loss'] = loss
        outputs['losses'] = losses
        outputs['reward'] = reward
        outputs['quality'] = quality
        outputs['delay'] = delay

    # get the loss for the actor (very hacky, need to re-compute a lot)
    if actor is not None:

        # get the highest reward
        if rescoring:
            re_index = reward.max(1)[1].data
        else:
            re_index = probs.data.new((batch_size, )).zero_().long()

        # loss for the model ---
        # top_model_loss = traj_loss.gather(1, re_index[:, None]).mean()

        # loss for the actor ---
        # prepare the data
        top_traj = traj.gather(1, re_index[:, None, None].expand(batch_size, 1, lx + ly)).squeeze(1)
        top_traj_mask = traj_mask.gather(1, re_index[:, None, None].expand(batch_size, 1, lx + ly)).squeeze(1)
        top_traj_index = torch.cumsum(top_traj, dim=1).long()
        top_traj_index = torch.cat([top_traj.new(top_traj.size(0), 1).zero_().long(), top_traj_index[:, :-1]], dim=1)
        top_traj_index = top_traj_index * top_traj_mask.long()
    
        # input side of the decoder
        top_inputs = inputs.gather(1, Variable(top_traj_index))
        top_input_masks = top_traj[:, None, :].expand(batch_size, lx + ly, lx + ly)
        _eye_matrix = torch.eye(lx + ly)[None, :, :].long()
        if top_input_masks.is_cuda: 
            _eye_matrix = _eye_matrix.cuda(top_input_masks.get_device())
        top_input_masks = (top_input_masks.long() | _eye_matrix).float()
        
        # source side
        top_src_index = torch.cumsum(1 - top_traj, dim=1)
        top_src_index = torch.cat([top_traj.new(batch_size, 1).zero_(), top_src_index[:, :-1]], dim=1)[:, :, None]  # first step ALL ZERO
        _range_matrix = torch.arange(lx)[None, None, :]
        if top_input_masks.is_cuda:
            _range_matrix = _range_matrix.cuda(top_input_masks.get_device())
        top_src_masks = ((top_src_index - _range_matrix) > 0).float() * source_masks[:, None, :]
     
        # run the model
        top_encoding = model.encoding(sources, source_masks)
        top_decoder_out = model(top_encoding, top_src_masks, top_inputs, top_input_masks, positions=Variable(top_traj_index.float()))
        
        if args.detach_decoder:
            top_actor_probs = actor(top_decoder_out.detach())
        else:
            top_actor_probs = actor(top_decoder_out)
        top_actor_loss = get_traj_loss(top_actor_probs[:, 1:], Variable(top_traj_mask[:, 1:]), Variable(top_traj[:, 1:])).mean()

        # void the possible loop
        top_targets = targets.gather(1, Variable(top_traj_index))
        top_target_mask = target_masks.gather(1, top_traj_index) * top_traj * top_traj_mask
        top_model_loss = model.cost(top_targets, top_target_mask, top_decoder_out)

        # get the loss
        outputs['model_loss'] = top_model_loss
        outputs['actor_loss'] = top_actor_loss

    # visualization
    if print_out:
        args.logger.info("{}: {}".format('[source]', model.output_decoding(("src", sources))[0]))
        args.logger.info("{}: {}".format('[target]', model.output_decoding(("trg", targets))[0]))
        args.logger.info(print_action(traj, traj_mask, losses, quality, delay, reward))
        args.logger.info('------------------------------------------------------------------')
        
    
    return outputs


def decode_model(args, model, actor, batch, print_out=False):
    print_seqs = ['[sources]', '[targets]', '[decoded]']
    print_acts = ['R', 'W']

    # pre-model information, prepare data and encoding
    inputs, input_masks, \
    targets, target_masks, \
    sources, source_masks, \
    encoding, batch_size = model.quick_prepare(batch)
    source_lens = source_masks.sum(dim=1)

    outputs, trajectories = model.decoder.simultaneous_greedy(actor, encoding, source_masks, target_masks)
    strings = [model.output_decoding(d) for d in [('src', sources), ('trg', targets), ('trg', outputs)]]
    filtered_string, filtered_trajtories = model.filter_actions(strings[2], trajectories)

    delay = torch.FloatTensor([get_delay_online(traj, sum(traj) / source_lens[k]) for k, traj in enumerate(filtered_trajtories)])
    quality = computeBLEU(strings[1], filtered_string)
    
    if print_out:
        args.logger.info("{}: {}".format('[sources]', strings[0][0]))
        args.logger.info("{}: {}".format('[targets]', strings[1][0]))
        args.logger.info("{}: {}".format('[decoded]', filtered_string[0])) 
        args.logger.info("{}: {}".format('[actions]', " ".join([print_acts[k] for k in filtered_trajtories[0]])))
        args.logger.info("{}: {}".format('[delayed]', "max_delay_score = {:.3f}".format(delay[0])))
        args.logger.info("{}: {}".format('[quality]', "sentence BLEU = {:.3f}".format(quality[0])))
        args.logger.info("===========================================================\n")

    return batch_size, delay, quality, strings[1], filtered_string


def train_inference(args, model, sampler, train, dev, save_path=None, maxsteps=None, writer=None):
    
    # save the model
    if save_path is None:
        save_path = args.model_name + '.Q.'
    # optimizer
    opt_sampler = torch.optim.Adam(sampler.parameters(), betas=(0.9, 0.98), eps=1e-9)
    
    best = Best(max, 'reward', 'quality', 'delay', 'i', model=sampler, opt=opt_sampler, path=save_path, gpu=args.gpu)
    train_metrics = Metrics('train', 'loss', 'reward', 'quality', 'delay')
    dev_metrics = Metrics('dev', 'loss', 'reward', 'quality', 'delay')
    progressbar = tqdm(total=args.eval_every, desc='start training.')

    for iters, batch in enumerate(train):
        model.eval()

        # --- saving --- 
        if iters % args.save_every == 0:
            args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
                torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(args.model_name, iters))

        # --- evaluation ---
        if iters % args.eval_every == 0:
            sampler.eval()
            progressbar.close()
            dev_metrics.reset()
            
            for dev_iters, dev_batch in enumerate(dev):
                printed = True if dev_iters < 10 else False
                dev_outputs = step(args, model, sampler, batch=dev_batch, n=10, stochastic=False, print_out=printed)
                reward, quality, delay = dev_outputs['reward'].mean(), dev_outputs['quality'].mean(), dev_outputs['delay'].mean()
                dev_metrics.accumulate(dev_outputs['batch_size'], dev_outputs['loss'], reward, quality, delay)
            
            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/quality', dev_metrics.quality, iters)
                writer.add_scalar('dev/delay', dev_metrics.delay, iters)
                writer.add_scalar('dev/reward', dev_metrics.reward, iters)
                writer.add_scalar('dev/loss', dev_metrics.loss, iters)

            if not args.debug:
                best.accumulate(reward, quality, delay, iters)
                args.logger.info('the best model is achieved at {}, reward={}'.format(best.i, best.reward))
            
            args.logger.info('model:' + save_path)
            args.logger.info(dev_metrics)

            # ---set-up a new progressor---
            progressbar = tqdm(total=args.eval_every, desc='start training.')

        if maxsteps is None:
            maxsteps = args.maximum_steps

        if iters > maxsteps:
            args.logger.info('reach the maximum updating steps.')
            break


        # --- training ---
        sampler.train()

        opt_sampler.zero_grad()
        outputs = step(args, model, sampler, batch)
        loss = outputs['loss']

        train_metrics.accumulate(outputs['batch_size'], 
                                outputs['loss'],
                                outputs['reward'].mean(),
                                outputs['quality'].mean(),
                                outputs['delay'].mean())
        
        loss.backward()
        opt_sampler.step()

        info = 'training step={}, loss={:.3f}, reward={:.3f}, quality={:.3f}, delay={:.3f}'.format(
            iters, train_metrics.loss, train_metrics.reward, train_metrics.quality, train_metrics.delay)
        progressbar.update(1)
        progressbar.set_description(info)
        train_metrics.reset()

        if np.isnan(np.sum(loss.data.cpu().numpy())):
            print('NaNaNaNaNaNa')
            import sys; sys.exit()


def train(args, train, dev, pre_model, model, actor, sampler, writer=None):
    
    # optimizer
    opt_sampler = torch.optim.Adam(sampler.parameters(), betas=(0.9, 0.98), eps=1e-9, lr=0.001)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
    opt_model = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    # best = Best(max, 'reward', 'i', model=model, opt=opt, path=save_path, gpu=args.gpu)
    train_metrics_p = Metrics('train', 'loss', 'actor_loss', 'model_loss')
    train_metrics_q = Metrics('train', 'loss', 'reward', 'quality', 'delay')
    dev_metrics = Metrics('dev', 'quality', 'delay', 'reward_f', 'quality_f', 'delay_f')
    progressbar = tqdm(total=args.eval_every, desc='start training.')

    # intervals
    p_steps = args.p_steps
    q_steps = args.q_steps
    updates = 0

    for iters, batch in enumerate(train):

         # --- evaluation ---
        if iters % args.eval_every == 0:

            model.eval(); actor.eval()
            progressbar.close()
            dev_metrics.reset()
            
            trg_outputs, dec_outputs = [], []
            for dev_iters, dev_batch in enumerate(dev):
                printed = True if dev_iters < 8 else False

                dev_outputs = step(args, model, sampler, batch=dev_batch, n=args.traj_size, stochastic=False, print_out=printed)
                reward, f_quality, f_delay = dev_outputs['reward'].mean(), dev_outputs['quality'].mean(), dev_outputs['delay'].mean()
                dev_batchsize, dev_delay, dev_quality, trgs, decs = decode_model(args, model, actor, dev_batch, printed)
                
                dev_metrics.accumulate(dev_batchsize, dev_quality.mean(), dev_delay.mean(), reward, f_quality, f_delay)
                trg_outputs += trgs
                dec_outputs += decs

            corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)

            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/gen_quality', dev_metrics.quality, iters)
                writer.add_scalar('dev/gen_delay', dev_metrics.delay, iters)
                writer.add_scalar('dev/corpus_bleu', corpus_bleu, iters)
                
            if dev_metrics is not None:
                args.logger.info(dev_metrics)
            args.logger.info("BLEU = {}".format(corpus_bleu))
  
            # ---set-up a new progressor---
            progressbar = tqdm(total=args.eval_every, desc='start training.')



        if iters % (p_steps + q_steps) < q_steps:
            
            # training the inference networks (REINFORCE)


            # --- training ---
            sampler.train(); pre_model.eval(); model.eval()

            if iters % args.inter_size == 0:
                opt_sampler.zero_grad()

            outputs = step(args, model, sampler, batch = batch, n=args.traj_size, stochastic=True)
            loss = outputs['loss']
            loss = loss / args.inter_size
            train_metrics_q.accumulate(outputs['batch_size'], 
                                    outputs['loss'],
                                    outputs['reward'].mean(),
                                    outputs['quality'].mean(),
                                    outputs['delay'].mean())
            loss.backward()

            if iters % args.inter_size == (args.inter_size - 1):
                opt_sampler.step()

            info = 'training step={}, loss={:.3f}, reward={:.3f}, quality={:.3f}, delay={:.3f}'.format(
                iters, train_metrics_q.loss, train_metrics_q.reward, train_metrics_q.quality, train_metrics_q.delay)
            progressbar.update(1)
            progressbar.set_description(info)
            train_metrics_q.reset()

        else:

            # training the model and the actor (Maximum Likelihood)
            

            # --- training ---
            sampler.eval(); pre_model.eval(); model.train(); actor.train()

            def get_learning_rate(i, lr0=0.1, disable=False):
                if not disable:
                    return lr0 * 10 / math.sqrt(args.d_model) * min(1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
                return 0.0001
            
            if iters % args.inter_size == 0:
                updates += 1
                lr = get_learning_rate(updates, disable=args.disable_lr_schedule)
                opt_model.param_groups[0]['lr'] = lr
                # opt_actor.param_groups[0]['lr'] = lr

                opt_model.zero_grad()
                opt_actor.zero_grad()


            outputs = step(args, model, sampler, actor, pre_model, batch=batch, n=args.traj_size, stochastic=False, rescoring=False)  # only use beam-search
            loss = outputs['model_loss'] + outputs['actor_loss']
            loss = loss / args.inter_size
            train_metrics_p.accumulate(outputs['batch_size'], loss * args.traj_size, outputs['actor_loss'], outputs['model_loss'])

                                    
            loss.backward()

            if iters % args.inter_size == (args.inter_size - 1):
                opt_actor.step()

                if not args.fix_model:
                    opt_model.step()

            info = 'training step={}, lr={:.6f}, loss={:.3f}, actor_loss={:.3f}, model_loss={:.3f}'.format(
                iters, lr, train_metrics_p.loss, train_metrics_p.actor_loss, train_metrics_p.model_loss)

            progressbar.update(1)
            progressbar.set_description(info)
            train_metrics_p.reset()


