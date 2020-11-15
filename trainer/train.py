#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torchtext
import os
import argparse
import logging
from torchtext.vocab import GloVe
from torch.utils.data import DataLoader
from trainer import Trainer

parser = argparse.ArgumentParser()

parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
opt = parser.parse_args()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# set up fields
TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,
                            use_vocab=True, init_token='<sos>',eos_token='<eos>')
LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
max_len = 512

def len_filter(example):
    return len(example.text) <= max_len
    
train = torchtext.data.TabularDataset(
        path = opt.train_path, format='tsv',
        fields = [('label',LABEL),('text',TEXT)],
        filter_pred = len_filter)

valid = torchtext.data.TabularDataset(
        path = opt.dev_path, format='tsv',
        fields = [('label', LABEL), ('text', TEXT)],
        filter_pred = len_filter)


TEXT.build_vocab(train, vectors=GloVe(name='6B', dim=300))
input_vocab = TEXT.vocab

# define model
# model = LSTM(#len(vocab), #embedding_vector, #max_len, #hidden_size, #bidirectional, #variable_length, #use_attn)
# model parameter initialziation
model = None
optimizer = None

# define loss and optimizer 
criterion = nn.CrossEntropyLoss()

# Train model
t = Trainer(loss=criterion, batch_size=4, 
            checkpoint_every=50, print_every=10,
            expt_dir=opt.expt_dir)
model = t.train(model, train, num_epochs=6,
                dev_data=valid, optimizer=optimizer)
