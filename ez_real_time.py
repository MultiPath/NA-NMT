import torch
import numpy as np
import math
import time

from torch.autograd import Variable
from tqdm import tqdm, trange
from model import Transformer, FastTransformer, SimultaneousTransformer, TINY, get_range
from utils import Metrics, Best, computeGLEU, computeBLEU, get_delay
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


def score_model(model, inputs, input_masks, targets, target_masks, encoding, traj_source_mask):
    traj_source_mask = traj_source_mask.transpose(2, 3)
    B, N, Ly, Lx = traj_source_mask.size()
    
    # expand and reshaping everything
    inputs = inputs[:, None, :].expand(B, N, Ly).contiguous().view(B * N, Ly)
    input_masks = input_masks[:, None, :].expand(B, N, Ly).contiguous().view(B * N, Ly)
    targets = targets[:, None, :].expand(B, N, Ly).contiguous().view(B * N, Ly)
    target_masks = target_masks[:, None, :].expand(B, N, Ly).contiguous().view(B * N, Ly)
    traj_source_mask = traj_source_mask.contiguous().view(B * N, Ly, Lx)
    
    for i in range(len(encoding)):
        d = encoding[i].size()[-1]
        encoding[i] = encoding[i][:, None, :, :].expand(B, N, Lx, d).contiguous().view(B * N, Lx, d)

    # compute the loss
    traj_decoder_out, traj_decoder_probs = model(encoding, traj_source_mask, inputs, input_masks, return_probs=True)
    cost, loss = model.batched_cost(targets, target_masks, traj_decoder_probs, batched=True)
    return traj_decoder_out, traj_decoder_probs, cost, loss.view(B, N)

def grid_path_loss(probs, path_mask, samples, entropy_weight=0.0):
    losses = -(torch.log(probs + TINY) * (samples - entropy_weight * probs) + 
               torch.log((1 - probs) + TINY) * ((1 - samples) - entropy_weight * (1 - probs)))
    losses = (losses * path_mask).sum(dim=-1).sum(dim=-1) / (path_mask.sum(dim=-1).sum(dim=-1) + TINY)
    return losses

def scorer(args, model, sampler=None, actor=None, pre_model=None,
            batch=None, n=10, stochastic=True, print_out=False):
    
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

    # run the inference-sampler
    probs, masks, samples, traj, traj_mask, path_mask, traj_src_mask = \
        sampler(encoder_out, decoder_out, source_masks, target_masks, n, stochastic, sample=True)
    path_mask = Variable(path_mask)

    # run the simultaneous translator
    encoding1 = model.encoding(sources, source_masks)
    encoder_traj = encoding1[-1]
    decoder_traj, decoder_probs, traj_cost, traj_loss = score_model(model, inputs, input_masks, targets, target_masks, encoding1, traj_src_mask)
    
    # get rewards: (1 - t) * quality + t * delay
    quality = traj_loss
    delay = get_delay(traj_src_mask, masks, type=args.delay_type)
    reward = -((1 - args.delay_weight) * quality + args.delay_weight * delay)

    # reward_shaping on each loss
    quality_shaped = shaping(quality, 1)
    delay_shaped = shaping(delay, 1)
    reward_shaped = -((1 - args.delay_weight) * quality_shaped + args.delay_weight * delay_shaped)

    # get the loss for the sampler
    losses = grid_path_loss(probs, path_mask, samples, 0.01)
    loss = (losses * reward_shaped.detach()).mean()
    
    # get the loss for the actor
    if actor is not None:

        # get the highest reward 
        re_index = reward.max(1)[1]

        # loss for the model
        max_traj_loss = traj_loss.gather(1, re_index[:, None]).mean()

        # loss for the actor
        decoder_traj = decoder_traj.view(-1, n, *decoder_traj.size()[1:])
        decoder_traj = decoder_traj.gather(1, re_index[:, None, None, None].expand(decoder_traj.size(0), 1, *decoder_traj.size()[2:])).squeeze(1)
        samples = samples.gather(1, re_index[:, None, None, None].expand(samples.size(0), 1, *samples.size()[2:])).squeeze(1)
        path_mask = path_mask.gather(1, re_index[:, None, None, None].expand(path_mask.size(0), 1, *path_mask.size()[2:])).squeeze(1)
        
        actor_probs  = actor.step(encoder_traj, decoder_traj)
        
        # real actor needs to read one word behind
        actor_samples = samples[:, 1:, :]
        actor_path_mask = path_mask[:, 1:, :]
        actor_probs = actor_probs[:, :-1, :]
        actor_losses = grid_path_loss(actor_probs, actor_path_mask, actor_samples, 0)
        actor_cost = actor_losses.mean()

    # visualization
    if print_out:
        args.logger.info("{}: {}".format('[source]', model.output_decoding(("src", sources))[0]))
        args.logger.info("{}: {}".format('[target]', model.output_decoding(("trg", targets))[0]))
        args.logger.info(print_action(traj, traj_mask, losses, quality, delay, reward))
        args.logger.info('------------------------------------------------------------------')

    outputs['traj_mask'] = traj_mask
    outputs['traj'] = traj
    outputs['batch_size'] = batch_size
    outputs['loss'] = loss
    outputs['losses'] = losses
    outputs['reward'] = reward
    outputs['quality'] = quality
    outputs['delay'] = delay

    if actor is not None: # training the real actor
        outputs['model_loss'] = max_traj_loss
        outputs['actor_loss'] = actor_cost

    return outputs


def q_actions(args, model, sampler, data, data_path):
    progressbar = tqdm(total=sum([1 for _ in data]), desc="start output R/W actions")
    model.eval()
    sampler.eval()

    output_f = open(data_path + '.Q1', 'w')
    for iters, batch in enumerate(data):
        outputs = scorer(args, model, sampler, batch, n=10, stochastic=False, print_out=False)
        indexs = outputs['reward'].sort(dim=-1, descending=True)[1][:, 0].data
        
        # ranking based on reward
        for b in range(outputs['traj'].size(0)):
            action_seq = []
            for i in range(outputs['traj'].size(2)):
                if (outputs['traj_mask'][b, indexs[b], i] > 0):
                    action_seq.append(str(int(outputs['traj'][b, indexs[b], i])))

            print(" ".join(action_seq))
        print('test decoding actions')
        break
    pass


def q_step(args, model, sampler, train, dev, save_path=None, maxsteps=None, writer=None):
    
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
                dev_outputs = scorer(args, model, sampler, dev_batch, n=10, stochastic=False, print_out=printed)
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
        outputs = scorer(args, model, sampler, batch)
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


def full_step(args, pre_model, model, actor, sampler, train, dev, writer=None):
    
    # optimizer
    opt_sampler = torch.optim.Adam(sampler.parameters(), betas=(0.9, 0.98), eps=1e-9)
    opt = torch.optim.Adam(list(model.parameters()) + list(actor.parameters()), betas=(0.9, 0.98), eps=1e-9)

    # best = Best(max, 'reward', 'i', model=model, opt=opt, path=save_path, gpu=args.gpu)
    train_metrics = Metrics('train', 'loss', 'reward', 'quality', 'delay')
    dev_metrics = Metrics('dev', 'loss', 'reward', 'quality', 'delay')
    progressbar = tqdm(total=args.eval_every, desc='start training.')

    for iters, batch in enumerate(train):
        # if iters > 5:
        #     break
        model.eval()

        # --- evaluation ---
        if False: #iters % args.eval_every == 0:
            sampler.eval()
            progressbar.close()
            dev_metrics.reset()
            
            for dev_iters, dev_batch in enumerate(dev):
                printed = True if dev_iters < 6 else False
                dev_outputs = scorer(args, model, sampler, dev_batch, n=10, stochastic=False, print_out=printed)

                dev_metrics.accumulate(dev_outputs['batch_size'], 
                                        dev_outputs['loss'],
                                        dev_outputs['reward'].mean(),
                                        dev_outputs['quality'].mean(),
                                        dev_outputs['delay'].mean())
            
            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/quality', dev_metrics.quality, iters)
                writer.add_scalar('dev/delay', dev_metrics.delay, iters)
                writer.add_scalar('dev/reward', dev_metrics.reward, iters)
                writer.add_scalar('dev/loss', dev_metrics.loss, iters)

            print(dev_metrics)

            # ---set-up a new progressor---
            progressbar = tqdm(total=args.eval_every, desc='start training.')


        # --- training ---
        sampler.eval()
        pre_model.eval()
        model.train()
        actor.train()

        opt.zero_grad()
        outputs = scorer(args, model, sampler, actor, pre_model, batch, args.traj_size, False)
        loss = outputs['model_loss'] + outputs['actor_loss']

        loss.backward()
        opt.step()

        print('ok')

        import sys; sys.exit()
        sampler.train()

        opt_sampler.zero_grad()
        outputs = scorer(args, model, sampler, batch)
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

