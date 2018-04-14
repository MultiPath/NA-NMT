# encoding: utf-8
import torch
import numpy as np
from torchtext import data
from torchtext import datasets

import revtok
import logging
import random
import argparse
import os
import copy
import sys
import math
import time

from ez_train import export, valid_model
from decode import decode_model
from model import Transformer, UniversalTransformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset, LazyParallelDataset, merge_cache
from utils import Metrics, Best, computeGLEU, computeBLEU
from time import gmtime, strftime
from tqdm import tqdm, trange
from torch.autograd import Variable


# all the hyper-parameters
parser = argparse.ArgumentParser(description='Train a Transformer-Like Model.')

# dataset settings
parser.add_argument('--data_prefix', type=str, default='/data0/data/transformer_data/')
parser.add_argument('--workspace_prefix', type=str, default='./')
parser.add_argument('--dataset',   type=str, default='iwslt', help='"the name of dataset"')
parser.add_argument('-s', '--src', type=str, default='ro',  help='meta-testing target language.')
parser.add_argument('-t', '--trg', type=str, default='en',  help='meta-testing target language.')
parser.add_argument('-a', '--aux', nargs='+', type=str,  default='es it pt fr',  help='meta-testing target language.')

parser.add_argument('--load_vocab',   action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--use_revtok',   action='store_true', help='use reversible tokenization')
parser.add_argument('--remove_eos',   action='store_true', help='possibly remove <eos> tokens for FastTransformer')
parser.add_argument('--test_set',     type=str, default=None,  help='which test set to use')
parser.add_argument('--max_len',      type=int, default=None,  help='limit the train set sentences to this many tokens')

# model basic settings
parser.add_argument('--prefix', type=str, default='[time]',      help='prefix to denote the model, nothing or [time]')
parser.add_argument('--params', type=str, default='james-iwslt', help='pamarater sets: james-iwslt, t2t-base, etc')

# model ablation settings
parser.add_argument('--causal_enc', action='store_true', help='use unidirectional encoder (useful for real-time translation)')
parser.add_argument('--causal',   action='store_true', help='use causal attention')
parser.add_argument('--diag',     action='store_true', help='ignore diagonal attention when doing self-attention.')
parser.add_argument('--use_wo',   action='store_true', help='use output weight matrix in multihead attention')
parser.add_argument('--share_embeddings',     action='store_true', help='share embeddings between encoder and decoder')
parser.add_argument('--positional_attention', action='store_true', help='incorporate positional information in key/value')

# running setting
parser.add_argument('--mode',    type=str, default='train',  help='train, test or build')
parser.add_argument('--gpu',     type=int, default=0,        help='GPU to use or -1 for CPU')
parser.add_argument('--seed',    type=int, default=19920206, help='seed for randomness')

# universal neural machine translation
parser.add_argument('--universal', action='store_true', help='enable embedding sharing in the universal space')
parser.add_argument('--inter_size', type=int, default=1, help='hack: inorder to increase the batch-size.')
parser.add_argument('--share_universal_embedding', action='store_true', help='share the embedding matrix with target. Currently only supports English.')
parser.add_argument('--finetune', action='store_true', help='add an action as finetuning. used for RO dataset.')
parser.add_argument('--universal_options', default='all', const='all', nargs='?',
                    choices=['no_use_universal', 'no_update_universal', 'no_update_self', 'no_update_encdec', 'all'], help='list servers, storage, or both (default: %(default)s)')
parser.add_argument('--meta_learning', action='store_true', help='meta-learning for low resource neural machine translation')

# meta-learning 
parser.add_argument('--no_meta_training',     action='store_true', help='no meta learning. directly training everything jointly.')
parser.add_argument('--sequential_learning',  action='store_true', help='default using a parallel training paradiam. However, it is another option to make training sequential.')
parser.add_argument('--valid_steps',   type=int, default=5,        help='repeating training for 5 epoches')
parser.add_argument('--inner_steps',   type=int, default=32,       help='every ** words for one meta-update (for default 160k)')
parser.add_argument('--outer_steps',   type=int, default=32,       help='every ** words for one meta-update (for default 160k)')
parser.add_argument('--eval-every',    type=int, default=1024,     help='run dev every')
parser.add_argument('--eval-every-examples', type=int, default=-1, help='alternative to eval every (batches)')
parser.add_argument('--save_every',    type=int, default=50000,   help='save the best checkpoint every 50k updates')
parser.add_argument('--maximum_steps', type=int, default=2000000, help='maximum steps you take to train a model')
parser.add_argument('--batch_size',    type=int, default=2048,    help='# of tokens processed per batch')
parser.add_argument('--optimizer',     type=str, default='Adam')
parser.add_argument('--disable_lr_schedule', action='store_true', help='disable the transformer-style learning rate')

parser.add_argument('--distillation', action='store_true', help='knowledge distillation at sequence level')
parser.add_argument('--finetuning',   action='store_true', help='knowledge distillation at word level')

# decoding
parser.add_argument('--length_ratio',  type=int,   default=2, help='maximum lengths of decoding')
parser.add_argument('--decode_mode',   type=str,   default='argmax', help='decoding mode: argmax, mean, sample, noisy, search')
parser.add_argument('--beam_size',     type=int,   default=1, help='beam-size used in Beamsearch, default using greedy decoding')
parser.add_argument('--f_size',        type=int,   default=1, help='heap size for sampling/searching in the fertility space')
parser.add_argument('--alpha',         type=float, default=1, help='length normalization weights')
parser.add_argument('--temperature',   type=float, default=1, help='smoothing temperature for noisy decodig')
parser.add_argument('--rerank_by_bleu', action='store_true', help='use the teacher model for reranking')

# model saving/reloading, output translations
parser.add_argument('--load_from',     type=str, default=None, help='load from checkpoint')
parser.add_argument('--resume',        action='store_true', help='when loading from the saved model, it resumes from that.')
parser.add_argument('--share_encoder', action='store_true', help='use teacher-encoder to initialize student')

parser.add_argument('--no_bpe',        action='store_true', help='output files without BPE')
parser.add_argument('--no_write',      action='store_true', help='do not write the decoding into the decoding files.')
parser.add_argument('--output_fer',    action='store_true', help='decoding and output fertilities')

# debugging
parser.add_argument('--debug',       action='store_true', help='debug mode: no saving or tensorboard')
parser.add_argument('--tensorboard', action='store_true', help='use TensorBoard')


args = parser.parse_args()
if args.prefix == '[time]':
    args.prefix = strftime("%m.%d_%H.%M.", gmtime())

# check the path
if not os.path.exists(args.workspace_prefix):
    os.mkdir(args.workspace_prefix)

def build_path(args, name):
    prefix = args.workspace_prefix
    pathname = os.path.join(prefix, name)
    if not os.path.exists(pathname):
        os.mkdir(pathname)
    args.__dict__.update({name + "_dir": pathname})
    return pathname

build_path(args, "models")
build_path(args, "runs")
build_path(args, "logs")


# setup logger settings
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

fh = logging.FileHandler('{}/log-{}.txt'.format(args.logs_dir, args.prefix))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)

# setup random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


# ----------------------------------------------------------------------------------------------------------------- #
data_prefix = args.data_prefix

# setup data-field
DataField = NormalField
TRG   = DataField(init_token='<init>', eos_token='<eos>', batch_first=True)
SRC   = DataField(batch_first=True) if not args.share_embeddings else TRG

# load universal pretrained embedding
if args.universal:
    U = torch.load(data_prefix + args.dataset + '/word_vec_tensor.pt')
    V = torch.load(data_prefix + args.dataset + '/word_vec_trg_tensor.pt')
    Freq = torch.load(data_prefix + args.dataset + '/freq_tensor.pt')[0]
    
    # simplest mask (not using frequency in the current stage.)
    Freq[:4] = 0
    Freq[4:] = 1

    if args.gpu > -1:
        U = U.cuda(args.gpu)
        V = V.cuda(args.gpu)
        Freq = Freq.cuda(args.gpu)
        
    args.__dict__.update({'U': U, 'V': V, 'Freq': Freq, 'unitok_size': V.size(0)})

# setup many datasets (need to manaually setup --- Meta-Learning settings.
logger.info('start loading the dataset')

if "europarl" in args.dataset:
    working_path = data_prefix + "{}/{}-{}/".format(args.dataset, args.src, args.trg)

    test_set = 'dev.tok'
    train_set = 'finetune.tok'

    train_data, dev_data = LazyParallelDataset.splits(path=working_path, train=train_set,
        validation=test_set, exts=('.src', '.trg'), fields=[('src', SRC), ('trg', TRG)])

    aux_data = [LazyParallelDataset(path=working_path + dataset, exts=('.src', '.trg'), 
                fields=[('src', SRC), ('trg', TRG)], lazy=True, max_len=110) for dataset in args.aux]
    decoding_path = working_path + '{}.' + args.src + '-' + args.trg + '.new'

else:
    raise NotImplementedError

logger.info('load dataset done..')


# build vocabularies
assert args.load_vocab and os.path.exists(data_prefix + '{}/vocab_{}.pt'.format(
        args.dataset, '{}-{}'.format(args.src, args.trg))), "meta-learning only works for pre-built vocabulary"

logger.info('load saved vocabulary.')
src_vocab, trg_vocab = torch.load(data_prefix + '{}/vocab_{}.pt'.format(args.dataset, '{}-{}'.format(args.src, args.trg)))
SRC.vocab = src_vocab
TRG.vocab = trg_vocab
logger.info('load done.')

args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

# build dynamic batching ---
def dyn_batch_with_padding(new, i, sofar):
    prev_max_len = sofar / (i - 1) if i > 1 else 0
    if args.distillation:
        return max(len(new.src), len(new.trg), len(new.dec), prev_max_len) * i
    else:
        return max(len(new.src), len(new.trg),  prev_max_len) * i

def dyn_batch_without_padding(new, i, sofar):
    if args.distillation:
        return sofar + max(len(new.src), len(new.trg), len(new.dec))
    else:
        return sofar + max(len(new.src), len(new.trg))


if args.batch_size == 1:  # speed-test: one sentence per batch.
    batch_size_fn = lambda new, count, sofar: count
else:
    batch_size_fn = dyn_batch_with_padding # dyn_batch_without_padding

train_real, dev_real = data.BucketIterator.splits(
    (train_data, dev_data), batch_sizes=(args.batch_size, args.batch_size), device=args.gpu, shuffle=False, 
    batch_size_fn=batch_size_fn, repeat=None if args.mode == 'train' else False)
aux_reals = [data.BucketIterator(dataset, batch_size=args.batch_size, device=args.gpu, train=True, batch_size_fn=batch_size_fn, shuffle=False)
            for dataset in aux_data]
logger.info("build the dataset. done!")


# ----------------------------------------------------------------------------------------------------------------- #
# model hyper-params:
logger.info('use default parameters of t2t-base')
hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
            'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
args.__dict__.update(hparams)

# ----------------------------------------------------------------------------------------------------------------- #
# show the arg:

hp_str = (f"{args.dataset}_subword_"
        f"{args.d_model}_{args.d_hidden}_{args.n_layers}_{args.n_heads}_"
        f"{args.drop_ratio:.3f}_{args.warmup}_{'universal_' if args.universal else ''}_meta")
logger.info(f'Starting with HPARAMS: {hp_str}')
model_name = args.models_dir + '/' + args.prefix + hp_str

# build the model
model = UniversalTransformer(SRC, TRG, args)

# logger.info(str(model))
if args.load_from is not None:
    with torch.cuda.device(args.gpu):
        model.load_state_dict(torch.load(args.models_dir + '/' + args.load_from + '.pt',
        map_location=lambda storage, loc: storage.cuda()))  # load the pretrained models.

# use cuda
if args.gpu > -1:
    model.cuda(args.gpu)

# additional information
args.__dict__.update({'model_name': model_name, 'hp_str': hp_str,  'logger': logger,  'n_lang': len(args.aux)})

# tensorboard writer
if args.tensorboard and (not args.debug):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('{}/{}'.format(args.runs_dir, args.prefix + args.hp_str))
else:
    writer = None

# show the arg:
arg_str = "args:\n"
for w in sorted(args.__dict__.keys()):
    if (w is not "U") and (w is not "V") and (w is not "Freq"):
        arg_str += "{}:\t{}\n".format(w, args.__dict__[w])
logger.info(arg_str)

# ----------------------------------------------------------------------------------------------------------------- #
#
# Starting Meta-Learning for Low-Resource Neural Machine Transaltion
#
# ----------------------------------------------------------------------------------------------------------------- #

# optimizer
if args.no_meta_training:
    meta_opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
else:  # meta-model only updates meta-parameters
    meta_opt = torch.optim.Adam([p for p in model.get_parameters(meta=True) if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
    
 # if resume training
if (args.load_from is not None) and (args.resume):
    with torch.cuda.device(args.gpu):   # very important.
        offset, opt_states = torch.load(args.models_dir + '/' + args.load_from + '.pt.states',
                                        map_location=lambda storage, loc: storage.cuda())
        meta_opt.load_state_dict(opt_states)
else:
    offset = 0

# ---- updates ------ #
iters = offset
eposides = 0
tokens = 0
time0 = time.time()

def get_learning_rate(i, lr0=0.1, disable=False):
    if not disable:
        return lr0 * 10 / math.sqrt(args.d_model) * min(1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
    return 0.00002


def inner_loop(args, data, model, weights, iters=0, save_diff=True, self_opt=None):
    model.train()
    data_loader, data_name = data
    progressbar = tqdm(total=args.inner_steps, desc='start training for {}'.format(data_name))

    
    model.load_fast_weights(weights)
    if self_opt is None:
        self_opt = torch.optim.Adam([p for p in model.get_parameters(meta=False) if p.requires_grad], betas=(0.9, 0.98), eps=1e-9) # reset the optimizer
    
    for i in range(args.inner_steps):                                                                                                
        self_opt.param_groups[0]['lr'] = get_learning_rate(iters + i + 1, disable=args.disable_lr_schedule)
        self_opt.zero_grad()
        loss_inner = 0
        bs_inner = 0
        for j in range(args.inter_size):
            train_batch = next(iter(data_loader))
            inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(train_batch)
            loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks)) / args.inter_size
            loss.backward()

            loss_inner = loss_inner + loss
            bs_inner = bs_inner + batch_size * max(inputs.size(1), targets.size(1))

        # update the fast-weights
        self_opt.step()
        info = '  Inner-loop[{}]: loss={:.3f}, lr={:.8f}, batch_size={}'.format(data_name, export(loss_inner), self_opt.param_groups[0]['lr'], bs_inner)
        progressbar.update(1)
        progressbar.set_description(info)

    progressbar.close()

    if save_diff:
        return model.save_fast_weights(weights=weights)  # fast-weights
    return model.save_fast_weights()


# training start..
best = Best(max, 'corpus_bleu', 'corpus_gleu', 'i', model=model, opt=meta_opt, path=args.model_name, gpu=args.gpu)
train_metrics = Metrics('train', 'loss', 'real', 'fake')
dev_metrics = Metrics('dev', 'loss', 'gleu', 'real_loss', 'fake_loss', 'distance', 'alter_loss', 'distance2', 'fertility_loss', 'corpus_gleu')

while True:

    # ----- saving the checkpoint ----- #
    if iters % args.save_every == 0:
        args.logger.info('save (back-up) checkpoints at iter={}'.format(iters))
        with torch.cuda.device(args.gpu):
            torch.save(best.model.state_dict(), '{}_iter={}.pt'.format(args.model_name, iters))
            torch.save([iters, best.opt.state_dict()], '{}_iter={}.pt.states'.format(args.model_name, iters))
       
    # ----- meta-validation ----- #
    if iters % args.eval_every == 0:
        dev_iters = iters
        with torch.cuda.device(args.gpu):
            weights = copy.deepcopy(model.save_fast_weights())  # --- initial params
            outs = copy.deepcopy(model.encoder.out.state_dict())

        fast_weights = weights
        self_opt = torch.optim.Adam([p for p in model.get_parameters(meta=False) if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
        corpus_bleu = -1
        corpus_gleu = -1

        outputs_data = valid_model(args, model, dev_real, dev_metrics, print_out=True)
        corpus_gleu0 = outputs_data['corpus_gleu']
        corpus_bleu0 = outputs_data['corpus_bleu']

        if args.tensorboard and (not args.debug):
            writer.add_scalar('dev/GLEU_corpus_', outputs_data['corpus_gleu'], dev_iters)
            writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], dev_iters)

        for j in range(args.valid_steps):
            args.logger.info("Fine-tuning step: {}".format(j))
            dev_metrics.reset()

            fast_weights = inner_loop(args, (train_real, "ro"), model, fast_weights, dev_iters, save_diff=False, self_opt=self_opt)
            outputs_data = valid_model(args, model, dev_real, dev_metrics, print_out=True)
            dev_iters += args.inner_steps
            
            if args.tensorboard and (not args.debug):
                writer.add_scalar('dev/GLEU_sentence_', dev_metrics.gleu, dev_iters)
                writer.add_scalar('dev/Loss', dev_metrics.loss, dev_iters)
                writer.add_scalar('dev/GLEU_corpus_', outputs_data['corpus_gleu'], dev_iters)
                writer.add_scalar('dev/BLEU_corpus_', outputs_data['corpus_bleu'], dev_iters)

            if outputs_data['corpus_gleu'] > corpus_gleu:
                corpus_gleu = outputs_data['corpus_gleu']

            if outputs_data['corpus_bleu'] > corpus_bleu:
                corpus_bleu = outputs_data['corpus_bleu']

            args.logger.info('model:' + args.prefix + args.hp_str + "\n")
            

        if args.tensorboard and (not args.debug):
            writer.add_scalar('dev/zero_shot_BLEU', corpus_bleu0, iters)
            writer.add_scalar('dev/fine_tune_BLEU', corpus_bleu, iters)

        if not args.debug:
            best.accumulate(corpus_bleu, corpus_gleu, iters)
            args.logger.info('the best model is achieved at {}, corpus GLEU={}, corpus BLEU={}'.format(
                best.i, best.corpus_gleu, best.corpus_bleu))

        args.logger.info('validation done.\n')
        model.load_fast_weights(weights)         # --- comming back to normal
        model.encoder.out.load_state_dict(outs)  # --- comming back to normal

    # ----- meta-training ------- #
    model.train()
    if iters > args.maximum_steps:
        args.logger.info('reach the maximum updating steps.')
        break

    # ----- inner-loop ------
    selected = random.randint(0, args.n_lang - 1)
    languages = range(args.n_lang) if not args.sequential_learning else [selected]

    if not args.no_meta_training:  # ----- only meta-learning requires inner-loop
        with torch.cuda.device(args.gpu):
            weights = copy.deepcopy(model.save_fast_weights())  # in case the data has been changed...
        all_fast_weights = []
        for j in languages:
            fast_weights = inner_loop(args, (aux_reals[j], args.aux[j]), model, weights, iters = iters, save_diff=True)
            all_fast_weights.append(fast_weights)   # save the increamentals
    
    
    
    # ------ outer-loop -----
    progressbar = tqdm(total=args.outer_steps, desc='start training')
    
    for k in range(args.outer_steps):
        meta_opt.param_groups[0]['lr'] = get_learning_rate(iters + k + 1, disable=args.disable_lr_schedule)
        meta_opt.zero_grad()
        loss_outer = 0
        bs_outter = 0

        if args.no_meta_training:
            assert not args.sequential_learning, "normal training used multiple languages"

        if not args.sequential_learning:
            for j in range(args.n_lang):
                meta_train_batch = next(iter(aux_reals[j]))
                if not args.no_meta_training:
                    model.load_fast_weights(all_fast_weights[j], weights)
                inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(meta_train_batch)
                loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks)) / args.n_lang
                loss.backward()   
                loss_outer = loss_outer + loss
                bs_outter = bs_outter + batch_size * max(inputs.size(1), targets.size(1))

        else:   # sequential training, only use one language.
            model.load_fast_weights(all_fast_weights[0], weights)
            for j in range(args.inter_size):
                meta_train_batch = next(iter(aux_reals[selected]))
                inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(meta_train_batch)
                loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks)) / args.inter_size
                loss.backward()   
                loss_outer = loss_outer + loss
                bs_outter = bs_outter + batch_size * max(inputs.size(1), targets.size(1))

        # update the meta-parameters
        if not args.no_meta_training:
            model.load_fast_weights(weights)
            meta_opt.step()
            with torch.cuda.device(args.gpu):
                weights = copy.deepcopy(model.save_fast_weights())
        else:
            meta_opt.step()

        info = 'Outer-loop (all): loss={:.3f}, lr={:.8f}, batch_size={}'.format(export(loss_outer), meta_opt.param_groups[0]['lr'], bs_outter)
        tokens = tokens + bs_outter

        if args.tensorboard and (not args.debug):
            writer.add_scalar('train/Loss', export(loss_outer), iters + k)
        
        progressbar.update(1)
        progressbar.set_description(info)
    
    progressbar.close()

    # ---- zero the self-embedding matrix
    if not args.no_meta_training:
        model.encoder.out.weight.data[4:, :].zero_() # ignore the first special tokens.

    iters = iters + args.outer_steps
    eposides = eposides + 1

    def hms(sec_elapsed):
        h = int(sec_elapsed / (60 * 60))
        m = int((sec_elapsed % (60 * 60)) / 60)
        s = sec_elapsed % 60.
        return "{}:{:>02}:{:>05.2f}".format(h, m, s)
    args.logger.info("Training {} tokens / {} batches / {} episodes, ends with: {}\n".format(tokens, iters, eposides, hms(time.time() - time0)))

args.logger.info('Done.')