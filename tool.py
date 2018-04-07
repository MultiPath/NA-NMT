import numpy as np
from torchtext import vocab
import torch
from collections import Counter

def process():

    vocab1 = torch.load('/data1/ywang/vocab_v2.pt')[0]
    word_vec = torch.load('/data1/ywang/word_vec_v2.pt')[0]
    vec_tensor = torch.Tensor(len(vocab1.stoi), 300).zero_()
    for key, value in word_vec.items():
        vec_tensor[vocab1.stoi[key]] = torch.from_numpy(np.squeeze(value))
    torch.save(vec_tensor, '/data1/ywang/word_vec_tensor.pt')


def process_trg():

    vocab1 = torch.load('/data1/ywang/vocab_v2.pt')[1]
    word_vec = torch.load('/data1/ywang/word_vec_v2.pt')[1]
    vec_tensor = torch.Tensor(len(vocab1.stoi), 300).zero_()
    for key, value in word_vec.items():
        vec_tensor[vocab1.stoi[key]] = torch.from_numpy(np.squeeze(value))
    torch.save(vec_tensor, '/data1/ywang/word_vec_trg_tensor.pt')




# process()  
# process_trg()
