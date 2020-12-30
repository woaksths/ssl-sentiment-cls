from __future__ import division
import logging
import os
import random
import time

import torch
import torchtext
from torch import optim

import torch.nn.functional as F
from optim import Optimizer
from evaluator import Evaluator
from util.vocab import itos, extract_lexicon, merge_lexicon
from util.checkpoint import Checkpoint
from util.lexicon_config import END_WORDS, STOP_WORDS
from trainer.EarlyStopping import EarlyStopping


class Trainer(object):
    """ This class helps in setting up a training framework in a supervised setting.
    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss: loss for training,
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of batches to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=None, batch_size=64,
                 random_seed=None, checkpoint_every=100, print_every=100):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss
        self.loss.name = 'cross_entropy'
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every
        
        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
                
        self.expt_dir = expt_dir
        
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)

        
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)
        self.input_vocab = None
        self.lexicon_stats = {0:{}, 1:{}}
        self.neg_lexicons = []
        self.pos_lexicons= []

        
    def _train_batch(self, input_variable, input_lengths, target_label, model):
        loss = self.loss
        # Forward propagation
#         logits, attn = model(input_variable, input_lengths, self.filtering_stoi_list)
        
        logits, attn = model(input_variable, input_lengths)
        lexicon = extract_lexicon(self.input_vocab, input_variable, target_label, logits, attn.squeeze(-1))
#         self.lexicon_stats = merge_lexicon(self.lexicon_stats, lexicon)        
        pos_wordsets = lexicon[0]
        neg_wordsets = lexicon[1]
        
        if len(neg_wordsets) > 0:
            self.neg_lexicons.extend(neg_wordsets)
        
        if len(pos_wordsets) > 0:
            self.pos_lexicons.extend(pos_wordsets)
        
        prob, indice = logits.max(1)
        # Get loss
        #loss.reset()
        self.optimizer.optimizer.zero_grad()
        
        loss = loss(logits, target_label)
        # Backward propagation
        loss.backward()
        self.optimizer.step()
        return loss

    
    def save_lexicons(self, path, lexicons):
#         print(path)
#         print(lexicons)
        if len(lexicons) >0:
            with open(path, 'w') as fw:
                for wordset in lexicons:
                    fw.write(' '.join(wordset) +'\n')
    
    
    def get_word_stats(self, lexicons):
        lexicon_dict ={}
        for word_set in lexicons:
            for word in word_set:
                if word in lexicon_dict:
                    lexicon_dict[word]+=1
                else:
                    lexicon_dict[word] =1
        return lexicon_dict
    
    
    def get_intersection(self, pos_lexicons, neg_lexicons, epoch):
        neg_dict = self.get_word_stats(neg_lexicons)
        pos_dict = self.get_word_stats(pos_lexicons)
        
        neg_dict = dict(sorted(neg_dict.items(), key=lambda x:x[1],reverse=True))
        pos_dict = dict(sorted(pos_dict.items(), key=lambda x:x[1], reverse=True))
        intersection = None
        
        if len(neg_dict) > 0 and len(pos_dict)>0:
            intersection = set(neg_dict.keys()) & set(pos_dict.keys())
            if len(intersection) >0:
#                 print('intersection', intersection, type(intersection))
                with open(self.lexicon_dir +'/intersection_epoch:{}'.format(epoch), 'w') as fw:
                    for word in intersection:
                        content = word + ' neg:{}, pos:{}'.format(neg_dict[word], pos_dict[word])
                        fw.write(content + '\n')
        return intersection
    
    
    def filter_common_word(self, common_words, lexicons, epoch):
        # filtering following words {END_WORDS, STOP_WORDS, common_words}
        neg_dict = self.get_word_stats(self.neg_lexicons)
        pos_dict = self.get_word_stats(self.pos_lexicons)
        filter_list = set(common_words) | set(END_WORDS) | set(STOP_WORDS)
        neg_dict = dict(sorted(neg_dict.items(), key=lambda x:x[1],reverse=True))
        pos_dict = dict(sorted(pos_dict.items(), key=lambda x:x[1], reverse=True))
        
        for word in filter_list:
            if word in neg_dict:
                del neg_dict[word]
            if word in pos_dict:
                del pos_dict[word]

        with open(self.lexicon_dir +'/pos_stats:{}'.format(epoch), 'w') as fw:
            for word in pos_dict:
                fw.write(word +' :' +str(pos_dict[word]) + '\n')
                
        with open(self.lexicon_dir +'/neg_stas:{}'.format(epoch), 'w') as fw:
            for word in neg_dict:
                fw.write(word +' :' +str(neg_dict[word]) + '\n')


    def filter_stoi(self):
        end_words_idx = [self.input_vocab.stoi[word] for word in END_WORDS]
        stop_words_idx = [self.input_vocab.stoi[word] for word in STOP_WORDS]
        return list(set(end_words_idx+stop_words_idx))

    
    
    def _train_epoches(self, data, model, n_epochs,
                       start_epoch, start_step, dev_data=None, test_data=None):
        log = self.logger
        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch
        
        device = torch.device('cuda:0') if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.text),
            device=device, repeat=False, shuffle=True)
        
        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs
        
        step = start_step
        step_elapsed = 0
        best_accuracy = 0
        
        early_stopping = EarlyStopping(patience = 10, verbose=True)
        
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))            
            batch_generator = batch_iterator.__iter__()
            
            # consuming seen batches from previous training
            for idx in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)
            
            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths = getattr(batch, 'text')
                target_variables = getattr(batch, 'label')
                loss  = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model)
                
                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    log_msg = 'Progress: %d%%, Train %s: %.4f' % (
                        step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

#             intersections = self.get_intersection(self.pos_lexicons, self.neg_lexicons, epoch)
#             if intersections is not None:
#                 self.filter_common_word(intersections, self.neg_lexicons, epoch)
#             self.save_lexicons(self.lexicon_dir +'/neg_epoch:{}'.format(epoch), self.neg_lexicons)
#             self.save_lexicons(self.lexicon_dir +'/pos_epoch:{}'.format(epoch), self.pos_lexicons)
            
            # reset neg/pos/intersection lexcions
            self.neg_lexicons = []
            self.pos_lexicons = []
            
            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            log_msg = "Finished epoch %d: Train %s: %.4f" % (epoch, self.loss.name, epoch_loss_avg)
            if dev_data is not None:
                model.eval()
                dev_loss, dev_accuracy = self.evaluator.evaluate(model, dev_data)
#                 self.optimizer.update(dev_loss, epoch)
                early_stopping(dev_loss, model, self.optimizer, epoch, step, self.input_vocab, self.expt_dir)
                if dev_accuracy > best_accuracy:
                    best_accuracy = dev_accuracy
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields['text'].vocab).save(self.expt_dir +'/best_accuracy')
                    print(self.expt_dir +'/best_accuracy')

                test_loss, test_acc = self.evaluator.evaluate(model, test_data)
                log_msg += ", Dev %s: %.4f, Accuracy: %.4f" % (self.loss.name, dev_loss, dev_accuracy)
                log_msg += ", test %s: %.4f, test Accuracy: %.4f" % (self.loss.name, test_loss, test_acc)
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)
            
            log.info(log_msg)
            if early_stopping.early_stop:
                print("Early Stopping")
                break
            
            
    def train(self, model, data, num_epochs=5,
              resume=False, dev_data=None, optimizer=None, test_data=None):
        """ Run training for a given model.
        Args:
            model: model to run training on, if `resume=True`, it would be
               overwritten by the model loaded from the latest checkpoint.
            data: dataset object to train on
            num_epochs (int, optional): number of epochs to run (default 5)
            resume(bool, optional): resume training with the latest checkpoint, (default False)
            dev_data: dev Dataset (default None)
            optimizer : optimizer for training
               (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))
        Returns:
            model: trained model.
        """
        
        # If training is set to resume
        if resume:
            latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
            resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
            model = resume_checkpoint.model
            self.optimizer = resume_checkpoint.optimizer

            # A walk around to set optimizing parameters properly
            resume_optim = self.optimizer.optimizer
            defaults = resume_optim.param_groups[0]
            defaults.pop('params', None)
            defaults.pop('initial_lr', None)
            self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

            start_epoch = resume_checkpoint.epoch
            step = resume_checkpoint.step
        else:
            start_epoch = 1
            step = 0
            if optimizer is None:
                optimizer = Optimizer(optim.Adam(model.parameters()), max_grad_norm=5)
            self.optimizer = optimizer
        
        
        self.input_vocab = data.fields['text'].vocab
        self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))
        self._train_epoches(data, model, num_epochs, start_epoch, step, dev_data=dev_data, test_data=test_data)
        return model
