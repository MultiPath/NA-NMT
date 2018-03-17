import torch
import numpy as np
import math
import time

from torch.autograd import Variable
from tqdm import tqdm, trange
from model import Transformer, FastTransformer, SimultaneousTransformer, TINY
from utils import Metrics, Best, computeGLEU, computeBLEU, get_delay, get_nll
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

def scorer(args, model, sampler, batch, n=10, stochastic=True, print_out=False):
    outputs = dict()

    # prepare the data
    inputs, input_masks, \
    targets, target_masks, \
    sources, source_masks, \
    encoding, batch_size = model.quick_prepare(batch)

    encoder_out = encoding[-1]
    decoder_out = model(encoding, source_masks, inputs, input_masks) 
    probs, masks, samples, traj, traj_mask, path_mask, traj_src_mask = sampler(encoder_out, decoder_out, source_masks, target_masks, n, stochastic)
   
    # prepare rewards
    quality = get_nll(model, inputs, input_masks, targets, target_masks, encoding, traj_src_mask)
    delay = get_delay(traj_src_mask, masks, type=args.delay_type)
    reward = -((1 - args.delay_weight) * quality + args.delay_weight * delay)

    # reward_shaping on each loss
    quality_shaped = shaping(quality, 1)
    delay_shaped = shaping(delay, 1)
    reward_shaped = -((1 - args.delay_weight) * quality_shaped + args.delay_weight * delay_shaped)

    path_mask = Variable(path_mask)
    losses = -(torch.log(probs + TINY) * (samples - 0.01 * probs) + torch.log(1 - probs + TINY) * (1 - samples))
    losses = (losses * path_mask).sum(dim=-1).sum(dim=-1) / (path_mask.sum(dim=-1).sum(dim=-1) + TINY)
    loss = (losses * reward_shaped.detach()).mean()
    
    # visualization
    if print_out:
        args.logger.info("{}: {}".format('[source]', model.output_decoding(("src", sources))[0]))
        args.logger.info("{}: {}".format('[target]', model.output_decoding(("trg", targets))[0]))
        args.logger.info(print_action(traj, traj_mask, losses, quality, delay, reward))
        args.logger.info('------------------------------------------------------------------')

    outputs['batch_size'] = batch_size
    outputs['loss'] = loss
    outputs['losses'] = losses
    outputs['reward'] = reward
    outputs['quality'] = quality
    outputs['delay'] = delay
    return outputs


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


def p_step(args, model, sampler, train, dev, writer=None):
    
    # optimizer
    opt_sampler = torch.optim.Adam(sampler.parameters(), betas=(0.9, 0.98), eps=1e-9)
    
    # best = Best(max, 'reward', 'i', model=model, opt=opt, path=save_path, gpu=args.gpu)
    train_metrics = Metrics('train', 'loss', 'reward', 'quality', 'delay')
    dev_metrics = Metrics('dev', 'loss', 'reward', 'quality', 'delay')
    progressbar = tqdm(total=args.eval_every, desc='start training.')

    for iters, batch in enumerate(train):
        # if iters > 5:
        #     break
        model.eval()

        # --- evaluation ---
        if iters % args.eval_every == 0:
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

