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

from ez_train import train_model
from decode import decode_model
from model import Transformer, FastTransformer, UniversalTransformer, INF, TINY, softmax
from utils import NormalField, NormalTranslationDataset, TripleTranslationDataset, ParallelDataset, LazyParallelDataset, merge_cache
from time import gmtime, strftime


# all the hyper-parameters
parser = argparse.ArgumentParser(description='Train a Transformer-Like Model.')

# dataset settings
parser.add_argument('--data_prefix', type=str, default='/data0/data/transformer_data/')
parser.add_argument('--workspace_prefix', type=str, default='./')
parser.add_argument('--dataset',     type=str, default='iwslt', help='"flickr" or "iwslt"')
parser.add_argument('--language',    type=str, default='ende',  help='a combination of two language markers to show the language pair.')
parser.add_argument('--data_group',  type=str, default=None,  help='dataset group')

parser.add_argument('--load_vocab',   action='store_true', help='load a pre-computed vocabulary')
parser.add_argument('--load_dataset', action='store_true', help='load a pre-processed dataset')
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
parser.add_argument('--finetune_dataset',  type=str, default=None)
parser.add_argument('--universal_options', default='all', const='all', nargs='?',
                    choices=['no_use_universal', 'no_update_universal', 'no_update_self', 'no_update_encdec', 'all'], help='list servers, storage, or both (default: %(default)s)')

# training
parser.add_argument('--eval-every',    type=int, default=1000,    help='run dev every')
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


# get the langauage pairs:
args.src = args.language[:2]  # source language
args.trg = args.language[2:]  # target language

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
DataField = data.ReversibleField if args.use_revtok else NormalField
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

# setup many datasets (need to manaually setup)
logger.info('start loading the dataset')

if args.dataset == 'iwslt':
    if args.test_set is None:
        args.test_set = 'IWSLT16.TED.tst2013'

    if args.data_group == 'test':
        train_data, dev_data = NormalTranslationDataset.splits(
        path=data_prefix + 'iwslt/en-de/', train='train.tags.en-de.bpe',
        validation='{}.en-de.bpe'.format(args.test_set), exts=('.{}'.format(args.src), '.{}'.format(args.trg)),
        fields=(SRC, TRG), load_dataset=args.load_dataset, prefix='normal')

    else:  # default dataset
        train_data, dev_data = ParallelDataset.splits(
        path=data_prefix + 'iwslt/en-de/', train='train.tags.en-de.bpe',
        validation='train.tags.en-de.bpe.dev', exts=('.en2', '.de2'),
        fields=[('src', SRC), ('trg', TRG)],
        load_dataset=args.load_dataset, prefix='ts')

    decoding_path = data_prefix + 'iwslt/en-de/{}.en-de.bpe.new'

elif "europarl" in args.dataset:
    if args.test_set is None:
        args.test_set = 'dev.tok'
    
    if args.finetune:
        if args.finetune_dataset is None:
            train_set = 'finetune.tok'
        else:
            train_set = args.finetune_dataset
    else:
        train_set = 'train.tok'

    working_path = data_prefix + "{}/{}-{}/".format(args.dataset, args.src, args.trg)
    train_data, dev_data = LazyParallelDataset.splits(path=working_path, train=train_set,
        validation=args.test_set, exts=('.src', '.trg'), fields=[('src', SRC), ('trg', TRG)],
        load_dataset=args.load_dataset, prefix='ts')

    # train_data, dev_data = ParallelDataset.splits(path=working_path, train='train.tok',
    #     validation=args.test_set, exts=('.src', '.trg'), fields=[('src', SRC), ('trg', TRG)],
    #     load_dataset=args.load_dataset, prefix='ts')
    decoding_path = working_path + '{}.' + args.src + '-' + args.trg + '.new'

else:
    raise NotImplementedError

logger.info('load dataset done..')

# build vocabularies
if args.load_vocab and os.path.exists(data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg))):

    logger.info('load saved vocabulary.')
    src_vocab, trg_vocab = torch.load(data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg)))
    SRC.vocab = src_vocab
    TRG.vocab = trg_vocab

    logger.info('load done.')


else:

    logger.info('save the vocabulary')
    if not args.share_embeddings:
        SRC.build_vocab(train_data, dev_data, max_size=50000)
    TRG.build_vocab(train_data, dev_data, max_size=50000)
    torch.save([SRC.vocab, TRG.vocab], data_prefix + '{}/vocab{}_{}.pt'.format(
        args.dataset, 'shared' if args.share_embeddings else '', '{}-{}'.format(args.src, args.trg)))

args.__dict__.update({'trg_vocab': len(TRG.vocab), 'src_vocab': len(SRC.vocab)})

# build alignments ---
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

# work around torchtext making it hard to share vocabs without sharing other field properties
if args.share_embeddings:
    SRC = copy.deepcopy(SRC)
    SRC.init_token = None
    SRC.eos_token = None
    train_data.fields['src'] = SRC
    dev_data.fields['src'] = SRC

if args.max_len is not None:
    train_data.examples = [ex for ex in train_data.examples if len(ex.trg) <= args.max_len]

if args.batch_size == 1:  # speed-test: one sentence per batch.
    batch_size_fn = lambda new, count, sofar: count
else:
    batch_size_fn = dyn_batch_with_padding # dyn_batch_without_padding


train_real, dev_real = data.BucketIterator.splits(
    (train_data, dev_data), batch_sizes=(args.batch_size, args.batch_size), device=args.gpu, shuffle=False, 
    batch_size_fn=batch_size_fn, repeat=None if args.mode == 'train' else False)
logger.info("build the dataset. done!")

# ----------------------------------------------------------------------------------------------------------------- #

# model hyper-params:
hparams = None
if args.dataset == 'iwslt':
    if args.params == 'james-iwslt':
        hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746} # ~32
    elif args.params == 'james-iwslt2':
        hparams = {'d_model': 278, 'd_hidden': 2048, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746} # ~32
    teacher_hparams = {'d_model': 278, 'd_hidden': 507, 'n_layers': 5,
                    'n_heads': 2, 'drop_ratio': 0.079, 'warmup': 746}


if hparams is None:
    logger.info('use default parameters of t2t-base')
    hparams = {'d_model': 512, 'd_hidden': 512, 'n_layers': 6,
                'n_heads': 8, 'drop_ratio': 0.1, 'warmup': 16000} # ~32
args.__dict__.update(hparams)

# ----------------------------------------------------------------------------------------------------------------- #
# show the arg:


hp_str = (f"{args.dataset}_subword_"
        f"{args.d_model}_{args.d_hidden}_{args.n_layers}_{args.n_heads}_"
        f"{args.drop_ratio:.3f}_{args.warmup}_{'uni_' if args.causal_enc else ''}")
logger.info(f'Starting with HPARAMS: {hp_str}')
model_name = args.models_dir + '/' + args.prefix + hp_str

# build the model
if not args.universal:
    model = Transformer(SRC, TRG, args)
else:
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

# show the arg:
arg_str = "args:\n"
for w in sorted(args.__dict__.keys()):
    if (w is not "U") and (w is not "V") and (w is not "Freq"):
        arg_str += "{}:\t{}\n".format(w, args.__dict__[w])
logger.info(arg_str)

if args.tensorboard and (not args.debug):
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('{}/{}'.format(args.runs_dir, args.prefix + args.hp_str))

# ----------------------------------------------------------------------------------------------------------------- #
if args.mode == 'train':
    logger.info('starting training')
    train_model(args, model, train_real, dev_real, writer=writer)

elif args.mode == 'test':
    logger.info('starting decoding from the pre-trained model, on the test set...')
    name_suffix = '{}_b={}_model_{}.txt'.format(args.decode_mode, args.beam_size, args.load_from)
    names = ['src.{}'.format(name_suffix), 'trg.{}'.format(name_suffix),'dec.{}'.format(name_suffix)]

    if args.model is FastTransformer:
        names += ['fer.{}'.format(name_suffix)]
    if args.rerank_by_bleu:
        teacher_model = None
    decode_model(args, model, dev_real, evaluate=True, decoding_path=decoding_path if not args.no_write else None, names=names)

logger.info("done.")
