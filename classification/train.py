#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torchtext
import os
import argparse
import logging
from torchtext.vocab import GloVe, Vocab
from torch.utils.data import DataLoader
import sys
from trainer import Trainer
from trainer import SSLTrainer
from model import LSTM 
from optim import Optimizer
from torch.optim.lr_scheduler import StepLR


parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--train_path_temp', action='store', dest='train_path_temp',
                    help='Path to train_path_temp')
parser.add_argument('--unlabeled_path', action='store', dest='unlabeled_path',
                    help='Path to unlabeled data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--is_ssl_train', action='store_true', dest='is_ssl_train', default=False,
                    help='Indicates if train way is based on semi-supervised learning')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# set up fields
TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,
                            use_vocab=True, sequential=True)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)
max_len = 64

def len_filter(example):
    return len(example.text) <= max_len

pseudo_train = torchtext.data.TabularDataset(
    path = opt.train_path_temp, format='tsv',
    fields = [('label',LABEL),('text',TEXT)])

train_data = torchtext.data.TabularDataset(
    path = opt.train_path, format='tsv',
    fields = [('label',LABEL),('text',TEXT)],
    filter_pred = len_filter)

unlabeled_data = torchtext.data.TabularDataset(
    path=opt.unlabeled_path, format='tsv',
    fields= [('text', TEXT)],
    filter_pred=len_filter)

valid_data = torchtext.data.TabularDataset(
    path = opt.dev_path, format='tsv',
    fields = [('label', LABEL), ('text', TEXT)],
    filter_pred = len_filter)

# build vocabulary based on training data and related words set with training data
TEXT.build_vocab(pseudo_train, vectors=GloVe(name='840B', dim=300))
input_vocab = TEXT.vocab

# define model
hidden_size = 128 # or 128, 256
model = LSTM(input_vocab.vectors, max_len, hidden_size,
             variable_lengths=True, bidirectional=True)

if torch.cuda.is_available():
    model.cuda()

# model parameter initialziation
for param in model.named_parameters():
    if 'embedding' in param[0]:
        continue
    param[1].data.uniform_(-0.08, 0.08)

# define loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = Optimizer(torch.optim.Adam(model.parameters()), max_grad_norm=5)
scheduler = StepLR(optimizer.optimizer, 1)
optimizer.set_scheduler(scheduler)

if torch.cuda.is_available():
    criterion.cuda()

# Train supervised model
t = Trainer(loss=criterion, batch_size=4,
            checkpoint_every=300, print_every=100,
            expt_dir=opt.expt_dir +'/supervised')

model = t.train(model, train_data, num_epochs=40,
                dev_data=valid_data, optimizer=optimizer)

# Train semi-supervised model(pseudo-labeling-way)
if opt.is_ssl_train == True:
    ssl_trainer = SSLTrainer(model=model, loss=criterion, 
                             batch_size=16, optimizer=optimizer, vocab=input_vocab,
                             expt_dir=opt.expt_dir + '/SSL', 
                             labeled_data= train_data, unlabeled_data=unlabeled_data,
                             text_field=TEXT, label_field=LABEL)
    ssl_trainer.train(num_epochs=20, dev_data=valid_data)
    