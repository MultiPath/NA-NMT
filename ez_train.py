import torch
import numpy as np
import math

from torch.autograd import Variable
from tqdm import tqdm, trange
from model import Transformer, FastTransformer
from utils import Metrics, Best, computeGLEU, computeBLEU

# helper functions
def register_nan_checks(m):
    def check_grad(module, grad_input, grad_output):
        if any(np.any(np.isnan(gi.data.cpu().numpy())) for gi in grad_input if gi is not None):
            print('NaN gradient in ' + type(module).__name__)
            1/0
    m.apply(lambda module: module.register_backward_hook(check_grad))

def export(x):
    try:
        with torch.cuda.device_of(x):
            return x.data.cpu().float().mean()
    except Exception:
        return 0

def devol(batch):
    new_batch = copy.copy(batch)
    new_batch.src = Variable(batch.src.data, volatile=True)
    return new_batch

tokenizer = lambda x: x.replace('@@ ', '').split()

def valid_model(args, model, dev, dev_metrics=None, distillation=False, print_out=False):
    print_seqs = ['[sources]', '[targets]', '[decoded]', '[fertili]', '[origind]']
    trg_outputs, dec_outputs = [], []
    outputs = {}

    model.eval()
    progressbar = tqdm(total=len([1 for _ in dev]), desc='start decoding for validation...')

    for j, dev_batch in enumerate(dev):
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks, \
        encoding, batch_size = model.quick_prepare(dev_batch, distillation)

        decoder_inputs, input_reorder, fertility_cost = inputs, None, None
        if type(model) is FastTransformer:
            decoder_inputs, input_reorder, decoder_masks, fertility_cost, pred_fertility = \
                model.prepare_initial(encoding, sources, source_masks, input_masks, None, mode='argmax')
        else:
            decoder_masks = input_masks

        decoding, out, probs = model(encoding, source_masks, decoder_inputs, decoder_masks, decoding=True, return_probs=True)
        dev_outputs = [('src', sources), ('trg', targets), ('trg', decoding)]
        if type(model) is FastTransformer:
            dev_outputs += [('src', input_reorder)]
        dev_outputs = [model.output_decoding(d) for d in  dev_outputs]
        gleu = computeGLEU(dev_outputs[2], dev_outputs[1], corpus=False, tokenizer=tokenizer)

        if (print_out and (j < 5)):
            for k, d in enumerate(dev_outputs):
                args.logger.info("{}: {}".format(print_seqs[k], d[0]))
            args.logger.info('------------------------------------------------------------------')

        trg_outputs += dev_outputs[1]
        dec_outputs += dev_outputs[2]

        if dev_metrics is not None:
            values = [0, gleu]
            if fertility_cost is not None:
                values += [fertility_cost]
            dev_metrics.accumulate(batch_size, *values)

        info = 'Validation: decoding step={}, gleu={:.3f}'.format(j + 1, export(gleu.mean()))
        progressbar.update(1)
        progressbar.set_description(info)
    
    progressbar.close()

    corpus_gleu = computeGLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
    corpus_bleu = computeBLEU(dec_outputs, trg_outputs, corpus=True, tokenizer=tokenizer)
    outputs['corpus_gleu'] = corpus_gleu
    outputs['corpus_bleu'] = corpus_bleu
    if dev_metrics is not None:
        args.logger.info(dev_metrics)

    args.logger.info("The dev-set corpus GLEU = {}".format(corpus_gleu))
    args.logger.info("The dev-set corpus BLEU = {}".format(corpus_bleu))
    return outputs


def train_model(args, model, train, dev, save_path=None, maxsteps=None, writer=None):

    # optimizer
    if args.optimizer == 'Adam':
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
    else:
        raise NotImplementedError

    # if resume training
    if (args.load_from is not None) and (args.resume):
        with torch.cuda.device(args.gpu):   # very important.
            offset, opt_states = torch.load(args.models_dir + '/' + args.load_from + '.pt.states',
                                            map_location=lambda storage, loc: storage.cuda())
            if not args.finetune:  # if finetune, do not have history
                opt.load_state_dict(opt_states)
    else:
        offset = 0

    # metrics
    if save_path is None:
        save_path = args.model_name

    args.eval_every *= args.inter_size


    best = Best(max, 'corpus_bleu', 'corpus_gleu', 'gleu', 'loss', 'i', model=model, opt=opt, path=save_path, gpu=args.gpu)
    train_metrics = Metrics('train', 'loss', 'real', 'fake')
    dev_metrics = Metrics('dev', 'loss', 'gleu', 'real_loss', 'fake_loss', 'distance', 'alter_loss', 'distance2', 'fertility_loss', 'corpus_gleu')
    progressbar = tqdm(total=args.eval_every, desc='start training.')
    examples = 0
    first_step = True
    loss_outer = 0

    for iters, batch in enumerate(train):

        iters += offset
        
        # --- saving --- #
        if iters % args.save_every == 0:
            args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
            with torch.cuda.device(args.gpu):
                torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
                torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(args.model_name, iters))


        # --- validation --- #
        if ((args.eval_every_examples == -1) and (iters % args.eval_every == 0)) \
            or ((args.eval_every_examples > 0) and (examples > args.eval_every_examples)) \
            or first_step:

            first_step = False

            if args.eval_every_examples > 0:
                examples = examples % args.eval_every_examples

            for dev_iters, dev_batch in enumerate(dev):

                progressbar.close()
                dev_metrics.reset()

                if args.distillation:
                    outputs_course = valid_model(args, model, dev, dev_metrics, distillation=True)

                outputs_data = valid_model(args, model, dev, None if args.distillation else dev_metrics, print_out=True)
                if args.tensorboard and (not args.debug):
                    writer.add_scalar('dev/GLEU_sentence_', dev_metrics.gleu, iters / args.inter_size)
                    writer.add_scalar('dev/Loss', dev_metrics.loss, iters / args.inter_size)
                    writer.add_scalar('dev/GLEU_corpus_', outputs_data['corpus_gleu'], iters / args.inter_size)
                    writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], iters / args.inter_size)


                if not args.debug:
                    best.accumulate(outputs_data['corpus_bleu'], outputs_data['corpus_gleu'], dev_metrics.gleu, dev_metrics.loss, iters / args.inter_size)
                    args.logger.info('the best model is achieved at {}, average greedy GLEU={}, corpus GLEU={}, corpus BLEU={}'.format(
                        best.i, best.gleu, best.corpus_gleu, best.corpus_bleu))
                args.logger.info('model:' + args.prefix + args.hp_str)

            # ---set-up a new progressor---
            progressbar = tqdm(total=args.eval_every, desc='start training.')


        if maxsteps is None:
            maxsteps = args.maximum_steps

        if iters > maxsteps:
            args.logger.info('reach the maximum updating steps.')
            break


        # --- training --- #
        model.train()
        def get_learning_rate(i, lr0=0.1, disable=False):
            if not disable:
                return lr0 * 10 / math.sqrt(args.d_model) * min(1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
            return 0.00002
        
        if iters % args.inter_size == 0:
            opt.param_groups[0]['lr'] = get_learning_rate(iters / args.inter_size + 1, disable=args.disable_lr_schedule)
            opt.zero_grad()
            loss_outer = 0

        # prepare the data
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks,\
        encoding, batch_size = model.quick_prepare(batch, args.distillation)
        input_reorder, fertility_cost, decoder_inputs = None, None, inputs

        examples += batch_size

        # Maximum Likelihood Training
        loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks)) / args.inter_size
        loss_outer = loss_outer + loss

        # accmulate the training metrics
        train_metrics.accumulate(batch_size, loss, print_iter=None)
        train_metrics.reset()

        loss.backward()
        
        if iters % args.inter_size == (args.inter_size - 1):

            if args.universal_options == 'no_update_encdec':
                for p in model.parameters():
                    if p is not model.encoder.uni_out.weight:
                        if p.grad is not None:
                            p.grad.detach_()
                            p.grad.zero_()

            opt.step()

            info = 'training step={}, loss={:.3f}, lr={:.8f}'.format(iters / args.inter_size, export(loss_outer), opt.param_groups[0]['lr'])
            if args.tensorboard and (not args.debug):
                writer.add_scalar('train/Loss', export(loss_outer), iters / args.inter_size)

            progressbar.update(1)
            progressbar.set_description(info)
