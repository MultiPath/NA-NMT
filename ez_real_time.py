import torch
import numpy as np
import math

from torch.autograd import Variable
from tqdm import tqdm, trange
from model import Transformer, FastTransformer, SimultaneousTransformer
from utils import Metrics, Best, computeGLEU, computeBLEU
from ez_train import *


def try_model(args, model, sampler, train, dev):
    if args.tensorboard and (not args.debug):
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('./runs/{}'.format(args.prefix + args.hp_str))

    for iters, batch in enumerate(train):

        model.train()

        # prepare the data
        inputs, input_masks, \
        targets, target_masks, \
        sources, source_masks, \
        encoding, batch_size = model.quick_prepare(batch)
        decoding = model(encoding, source_masks, inputs, input_masks)

        print(encoding.size())
        print(decoding.size())
        print(sampler(encoding, decoding).size())

        break