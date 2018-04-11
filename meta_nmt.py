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

from ez_train import export
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
parser.add_argument('--dataset',     type=str, default='iwslt', help='"the name of dataset"')
parser.add_argument('-s', '--src',  type=str, default='ro',  help='meta-testing target language.')
parser.add_argument('-t', '--trg',  type=str, default='en',  help='meta-testing target language.')
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

# training
parser.add_argument('--eval-every',    type=int, default=1000,    help='run dev every')
parser.add_argument('--meta-eval-every', type=int, default=32,   help='every ** words for one meta-update (for default 160k)')
parser.add_argument('--eval-every-examples', type=int, default=-1, help='alternative to eval every (batches)')
parser.add_argument('--save_every',    type=int, default=50000,   help='save the best checkpoint every 50k updates')
parser.add_argument('--maximum_steps', type=int, default=1000000, help='maximum steps you take to train a model')
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

    aux_data = [LazyParallelDataset(path=working_path + dataset, exts=('.src', '.trg'), fields=[('src', SRC), ('trg', TRG)], lazy=True)
                for dataset in args.aux]
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
    batch_size_fn = dyn_batch_without_padding


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
args.__dict__.update({'model_name': model_name, 'hp_str': hp_str,  'logger': logger})

# tensorboard writer
if args.tensorboard and (not args.debug):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('{}/{}'.format(args.runs_dir, args.prefix + args.hp_str))

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
if args.optimizer == 'Adam':
    meta_opt = torch.optim.Adam([p for p in model.get_parameters(meta=True) if p.requires_grad], betas=(0.9, 0.98), eps=1e-9)
else:
    raise NotImplementedError

# ---- updates ------
args.eval_every *= args.inter_size
args.meta_eval_every *= args.inter_size

iters = 0

# ---- outer-loop ---
while True:
    model.train()
    def get_learning_rate(i, lr0=0.1, disable=False):
        if not disable:
            return lr0 * 10 / math.sqrt(args.d_model) * min(1 / math.sqrt(i), i / (args.warmup * math.sqrt(args.warmup)))
        return 0.00002 
    lr0 = get_learning_rate(iters / args.inter_size + 1)

    # saving the checkpoint #
    if iters % args.save_every == 0:
        pass
    
    # ----- inner-loop ------
    meta_param = copy.deepcopy(model.save_self_parameters())  # in case the data has been changed...
    self_params = []
    for j in range(len(aux_data)):
        progressbar = tqdm(total=args.meta_eval_every, desc='start training for {}'.format(args.aux[j]))

        # reset the optimizer
        model.load_self_parameters(meta_param)
        self_opt = torch.optim.Adam([p for p in model.get_parameters(meta=False) if p.requires_grad], betas=(0.9, 0.98), eps=1e-9, lr=lr0)
        for i, train_batch in enumerate(aux_reals[j]):
            if i % args.inter_size == 0:
                self_opt.param_groups[0]['lr'] = get_learning_rate(iters + i / args.inter_size + 1, disable=args.disable_lr_schedule)
                self_opt.zero_grad()
            
            inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(train_batch)
            loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks)) / args.inter_size
            loss.backward()

            if i % args.inter_size == (args.inter_size - 1):
                self_opt.step()

            info = '     Inner-loop[{}]: training step={}, loss={:.3f}, lr={:.8f}'.format(args.aux[j], i, export(loss * args.inter_size), self_opt.param_groups[0]['lr'])
            progressbar.update(1)
            progressbar.set_description(info)

            if i == args.meta_eval_every:
                break
        progressbar.close()
        self_params.append(model.save_self_parameters(meta_param=meta_param))   # save the increamentals
    
    # ------ outer-loop -------
    progressbar = tqdm(total=args.meta_eval_every, desc='start training for {}'.format(args.aux[j]))
    for k in range(args.meta_eval_every):
        meta_opt.param_groups[0]['lr'] = get_learning_rate(iters + k + 1, disable=args.disable_lr_schedule)
        loss_outer = 0
        for j in range(len(aux_data)):
            meta_train_batch = next(iter(aux_reals[j]))
            model.load_self_parameters(self_params[j], meta_param)
            inputs, input_masks, targets, target_masks, sources, source_masks, encoding, batch_size = model.quick_prepare(meta_train_batch)
            
            loss = model.cost(targets, target_masks, out=model(encoding, source_masks, inputs, input_masks)) / len(aux_data)
            loss.backward()   
            loss_outer = loss_outer + loss
        
        # update the meta-parameters
        model.load_self_parameters(meta_param)
        meta_opt.step()
        meta_param = copy.deepcopy(model.save_self_parameters())

        info = 'Outer-loop (all): training step={}, loss={:.3f}, lr={:.8f}'.format(iters + k, export(loss_outer), self_opt.param_groups[0]['lr'])
        progressbar.update(1)
        progressbar.set_description(info)
    progressbar.close()
    
    iters = iters + args.meta_eval_every
    print('done')
    break
