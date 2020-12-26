#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torchtext
import os
import random
import argparse
import logging
from torchtext.vocab import GloVe, Vocab
from torch.utils.data import DataLoader
import sys
from trainer import Trainer
from trainer import SSLTrainer
from model import LSTM, Classifier_Attention_LSTM
from optim import Optimizer
from torch.optim.lr_scheduler import StepLR
from evaluator import Evaluator
from util.checkpoint import Checkpoint
from util.eda import *
from util.dataset import *
from util.dataset import get_golden_lexicons, get_annotated_word_tag

parser = argparse.ArgumentParser()

###################################################################
### arugment for supervised learning
###################################################################
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--lexicon_dir', action='store', dest='lexicon_dir', default='./generated_lexicons',
                    help='Path to generated lexcion directory.')
parser.add_argument('--sampled_dataset_dir', action='store', dest='sampled_dataset_dir', default='./sampled_dataset',
                    help='Path to sampled dataset directory.')

##################################################################
### argument for semi supervised learning
##################################################################
parser.add_argument('--is_ssl_train', action='store_true', dest='is_ssl_train', default=False,
                    help='Indicates if train way is based on semi-supervised learning')
parser.add_argument('--annotated_lexicon', action='store', dest='annotated_lexicon', 
                    help='golden lexicon about extremely small training dataset')
parser.add_argument('--labeled_data', action='store', dest='labeled_data', 
                    help='extremely small training dataset')
parser.add_argument('--unlabeled_data', action='store', dest='unlabeled_data',
                    help='a vast of unlabeled dataset')
parser.add_argument('--dev_data', action='store', dest='dev_data', 
                    help='dev data set for labeled_data')
parser.add_argument('--test_data', action='store', dest='test_data',
                    help='test data set')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# config dataset
max_len = 300
sample_num_per_class = 10
class_num = 2
num_aug = 0

# set up fields
TEXT = torchtext.data.Field(lower=True, include_lengths=True, batch_first=True,
                            use_vocab=True, sequential=True)
LABEL = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)

if not opt.is_ssl_train:  
    # load dataset
    
    total_dataset = get_dataset(opt.train_path)
    test_dataset = get_dataset(opt.dev_path)

    # define sampled dataset directory 
    sampled_dataset_dir = opt.sampled_dataset_dir
    if not os.path.isabs(sampled_dataset_dir):
        sampled_dataset_dir = os.path.join(os.getcwd(), sampled_dataset_dir)
    if not os.path.exists(sampled_dataset_dir):
        os.makedirs(sampled_dataset_dir)

    # make initital dataset (train, dev, unlabeled)
    train_data, dev_data, unlabeled_data = sampling_initial_dataset(total_dataset, class_num, sample_num_per_class)
    augmented_train = gen_augmented_dataset(train_data, num_aug)
    augmented_dev = gen_augmented_dataset(dev_data, 0)

    write_sampled_dataset(sampled_dataset_dir +'/augmented_train.txt', augmented_train)
    write_sampled_dataset(sampled_dataset_dir +'/augmented_dev.txt', augmented_dev)
    write_sampled_dataset(sampled_dataset_dir +'/labeled_data.txt', train_data)
    write_sampled_dataset(sampled_dataset_dir +'/dev.txt', dev_data)
    write_sampled_dataset(sampled_dataset_dir +'/unlabeled_data.txt', unlabeled_data)
    
    # experiment1: augmented_train으로 했을 때와 아닐 때 
    train_data = augmented_train 
    dev_data = augmented_dev    

    train_examples = examples_from_dataset(train_data, max_len)
    dev_examples = examples_from_dataset(dev_data, max_len)
    test_examples = examples_from_dataset(test_dataset,  max_len)
    unlabeled_examples = examples_from_dataset(unlabeled_data, max_len)

    glove = GloVe(name='840B', dim=300)
    glove_example = glove_to_example(glove.stoi.keys())
    total_examples = train_examples + unlabeled_examples + dev_examples + [glove_example]
    
    total_data = torchtext.data.Dataset(total_examples, fields=[('text', TEXT), ('label', LABEL)])
    train_data = torchtext.data.Dataset(train_examples, fields=[('text', TEXT), ('label', LABEL)])
    dev_data = torchtext.data.Dataset(dev_examples, fields=[('text', TEXT), ('label', LABEL)])
    test_data = torchtext.data.Dataset(test_examples, fields=[('text', TEXT), ('label', LABEL)])
    
    TEXT.build_vocab(total_data, vectors=glove)
    input_vocab = TEXT.vocab
    # define model
    hidden_size = 128 # or 128, 256
    model = LSTM(input_vocab.vectors, max_len, hidden_size,
                 variable_lengths=True, bidirectional=False)
#     model = Classifier_Attention_LSTM(input_vocab.vectors, 2)

    if torch.cuda.is_available():
        model.cuda()

    # model parameter initialziation
    for param in model.parameters():
        param.data.uniform_(-0.08, 0.08)

    # define loss and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = Optimizer(torch.optim.Adam(model.parameters(), lr=0.001),max_grad_norm=5) # lr:2e-5
    scheduler = StepLR(optimizer.optimizer, 1)
    optimizer.set_scheduler(scheduler)

    if torch.cuda.is_available():
        criterion.cuda()

    # Train supervised model
    t = Trainer(loss=criterion, batch_size=128,
                checkpoint_every= 300, print_every=100,
                expt_dir=opt.expt_dir +'/supervised', lexicon_dir = opt.lexicon_dir)
    model, lexicon = t.train(model, train_data, num_epochs=50,
                             dev_data=dev_data, optimizer=optimizer, test_data=test_data)

    evaluator = Evaluator(loss=criterion, batch_size=64)
    loss, accuracy = evaluator.evaluate(model, test_data)
    print(model)
    print('supervised model::: loss > {} accuracy{}'.format(loss, accuracy))
else:
    assert opt.load_checkpoint is not None
    assert opt.unlabeled_data is not None 
    assert opt.labeled_data is not None
    assert opt.dev_data is not None
    assert opt.test_data is not None
    assert opt.annotated_lexicon is not None

    outputs_dir = opt.labeled_data.split('/')[0]

    # load sampled dataset and golden lexicon
    unlabeled_data = get_dataset(opt.unlabeled_data)
    labeled_data = get_dataset(opt.labeled_data)
    dev_data = get_dataset(opt.dev_data)
    test_data = get_dataset(opt.test_data)
    
    
    labled_examples = examples_from_dataset(labeled_data, max_len)
    unlabeled_examples = examples_from_dataset(unlabeled_data, max_len)
    dev_examples = examples_from_dataset(dev_data,  max_len)
    test_examples = examples_from_dataset(test_data,  max_len)

    labeled_dataset = torchtext.data.Dataset(labled_examples, fields=[('text', TEXT), ('label', LABEL)])
    unlabeled_dataset = torchtext.data.Dataset(unlabeled_examples, fields=[('text', TEXT), ('label', LABEL)])
    dev_dataset = torchtext.data.Dataset(dev_examples, fields=[('text', TEXT), ('label', LABEL)])
    test_dataset = torchtext.data.Dataset(test_examples, fields=[('text', TEXT), ('label', LABEL)])
#     golden_lexicons = get_golden_lexicons(opt.annotated_lexicon)
    golden_lexicons = get_annotated_word_tag(opt.annotated_lexicon, class_num=2)
    
    # load best checkpoint 
    checkpoint = Checkpoint.get_latest_checkpoint(opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint)
    input_vocab = checkpoint.input_vocab
    model = checkpoint.model
    TEXT.vocab = input_vocab
    
    # config 
    criterion = nn.CrossEntropyLoss()
    optimizer = checkpoint.optimizer
    resume_optim  = checkpoint.optimizer.optimizer
    defaults = resume_optim.param_groups[0]
    defaults.pop('params', None)
    defaults.pop('initial_lr', None)
    
    optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)
    
    evaluator = Evaluator(loss=criterion, batch_size=64)
    loss, accuracy = evaluator.evaluate(checkpoint.model, test_dataset)
    print(model)
    # labeled_dataset, unlabeled_dataset, dev_dataset, test_dataset, golden_lexicons
    print('supervised model::: loss > {} accuracy{}'.format(loss, accuracy))
    
    # Train semi-supervised model(pseudo-labeling-way)
    ssl_trainer = SSLTrainer(model=model, loss=criterion, 
                             batch_size=16, optimizer=optimizer,
                             vocab=input_vocab, expt_dir=opt.expt_dir + '/SSL', 
                             outputs_dir = outputs_dir, annotated_lexicons=golden_lexicons, 
                             labeled_data= labeled_dataset, unlabeled_data=unlabeled_dataset,
                             text_field=TEXT, label_field=LABEL)
    ssl_trainer.train(num_epochs=40, dev_data=dev_dataset, test_data=test_dataset)
    1/0