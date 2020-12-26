import random
from trainer import Trainer
from evaluator import Evaluator
import os
import torchtext
import torch
import logging
from util.vocab import itos, extract_lexicon, merge_lexicon, get_words_by_attn
from util.lexicon_config import STOP_WORDS, END_WORDS, NEGATOR

import torch.nn.functional as F
from util.augment import augment_antonym_lexicons, gen_reverse_sent, gen_reverse_sent_with_lexicons
from util.lexicon_utils import get_senti_lexicon


class SSLTrainer(object):
    def __init__(self, model=None, loss=None, batch_size=64, 
                 optimizer=None, vocab=None, expt_dir=None,
                 annotated_lexicons=None, labeled_data=None, unlabeled_data=None,
                 text_field=None, label_field=None, outputs_dir=None):
        
        self._trainer= "Semi supervised Trainer"
        self.model = model
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = optimizer

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        self.outputs_dir = outputs_dir
        
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        
        self.batch_size = batch_size
        self.input_vocab = vocab
        # field
        self.TEXT_field = text_field
        self.LABEL_field = label_field
        self.logger = logging.getLogger(__name__)
        self.dev_acc = None
        
        self.golden_lexicons_with_tag = annotated_lexicons
        
        # augment golden_lexicons with antinom
        #self.golden_lexicons =  augment_antonym_lexicons(self.golden_lexicons_with_tag)
        self.golden_lexicons = get_senti_lexicon()
        self.attn_lexicons = {0:{}, 1:{}}
        
        # dataset
        self.unlabeled_data = [( ' '.join(example.text).strip(), example.label) for example in unlabeled_data.examples]
        self.unlabeled_data = list(set(self.unlabeled_data))
        self.unlabeled_text = [data[0].strip() for data in self.unlabeled_data]
        self.unlabeled_target = [data[1] for data in self.unlabeled_data]
        self.labeled_examples = labeled_data.examples # type list 
        self.unlabeled_examples = [torchtext.data.Example.fromlist((text, label),[('text', self.TEXT_field), ('label', self.LABEL_field)]) for (text, label) in zip(self.unlabeled_text, self.unlabeled_target)]

        # Threshold
        self.matching_threshold = 7
        self.model_threshold = 0.9
        
        # Matching result
        self.matching_cnt = 0
        self.matching_correct = 0
    
        self.device = None

        
    def save_labeled_result(self, path, labeled_result):
        with open(path, 'w') as fw:
            for text, label in labeled_result:
                fw.write(str(label) +'\t' + text + '\n')

                

    def update_dataset(self, labeled_dataset):
        print('INFO. # num of current training dataset', len(self.labeled_examples))
        print('INFO. # num of current unlabeled dataset', len(self.unlabeled_text))
        print('INFO. # num of new labeled dataset', len(labeled_dataset))
#         print('INFO. # unlabeled dataset accuracy',self.matching_correct, self.matching_cnt ,float(self.matching_correct/self.matching_cnt))
        print('INFO. # unlabeled dataset accuracy',self.matching_correct, self.matching_cnt)
        
        self.matching_correct = 0 
        self.matching_cnt = 0
        new_labeled_examples = []
        for idx, (text, label) in  enumerate(labeled_dataset):
            text = text.strip()
            label = str(label)
            # remove labeled text from unlabeled_text
            if text in self.unlabeled_text:
                index = self.unlabeled_text.index(text)
                
                self.unlabeled_target.pop(index)
                self.unlabeled_text.pop(index)
                # append labeled text into labeled_set    
                example = torchtext.data.Example.fromlist((text, label),[('text', self.TEXT_field),
                                                                         ('label', self.LABEL_field)])
                self.labeled_examples.append(example)
                new_labeled_examples.append(example)
            else:
                print(text)
                print(label)
                1/0
        # update unlabeled examples
        unlabeled_examples = [torchtext.data.Example.fromlist((text, label),[('text', self.TEXT_field), ('label', self.LABEL_field)]) for (text, label) in zip(self.unlabeled_text, self.unlabeled_target)]
        self.unlabeled_examples = unlabeled_examples
        print('INFO. # num of remain unlabeled data', len(self.unlabeled_examples))
        print('INFO. # updated labeled data ', len(self.labeled_examples))
        print('*'*100)
        return new_labeled_examples
        
        
    def match_by_lexicon(self, text, golden_lexicons):
        match_count = {label:0 for label in golden_lexicons.keys()}
        matched_tokens = []
        token_list = text.split(' ')
        for idx, word in enumerate(token_list):
            for label in golden_lexicons:
                if word in golden_lexicons[label]:
                    has_negator = False 
                    if idx-2 >=0 and token_list[idx-2] in NEGATOR or token_list[idx-1] in NEGATOR:
                        label = 1 - label
                        has_negator =True
                    match_count[label] += 1
                    if has_negator:
                        matched_tokens.append(token_list[idx-2]+' '+token_list[idx-1]+' '+ word)
                    else:
                        matched_tokens.append(word)
        return match_count, matched_tokens

    
    
    def pseudo_labeling(self, input_vars, text_list, logits, attns, targets):
        pseudo_labels = []
        confidences, predicted_labels = logits.max(1)
        
        for idx, (input_var, text, label, prob, attn, target) in enumerate(zip(input_vars, text_list, predicted_labels, confidences, attns,targets)):
            matching_result, matched_tokens = self.match_by_lexicon(text, self.golden_lexicons)
            model_pred = label.item()
            lexicon_pred = None
            pred_label = None
            
            if self.dev_acc is None:
                if matching_result[0] > matching_result[1] and  matching_result[0] >= self.matching_threshold and matching_result[1] ==0: 
                    lexicon_pred = 0
                elif matching_result[0] < matching_result[1] and  matching_result[1] >= self.matching_threshold and matching_result[0] == 0:
                    lexicon_pred = 1
            elif prob >= self.model_threshold:
                if matching_result[0] > matching_result[1] and matching_result[0]>self.matching_threshold:
                    lexicon_pred = 0
                elif matching_result[0] < matching_result[1] and matching_result[1]>self.matching_threshold:
                    lexicon_pred = 1
            
            if lexicon_pred == model_pred:
                pred_label = lexicon_pred
            else:
                pred_label = None
                
            if pred_label is not None:
                self.matching_cnt += 1
                pseudo_labels.append((text, pred_label))
                
                if pred_label == target.item():
                    self.matching_correct += 1
                    
                _, attn_high_indices = attn.topk(k=5, dim=-1)
                attn_high_words=  get_words_by_attn(self.input_vocab, input_var, attn_high_indices)
                
                for word in attn_high_words:
                    if word in STOP_WORDS or word in END_WORDS:
                        continue
                    if word in self.attn_lexicons[pred_label]: 
                        self.attn_lexicons[pred_label][word] +=1
                    else:
                        self.attn_lexicons[pred_label][word] = 1
                        
                self.attn_lexicons[0] = dict(sorted(self.attn_lexicons[0].items(), key=lambda x:x[1],reverse=True))
                self.attn_lexicons[1] = dict(sorted(self.attn_lexicons[1].items(), key=lambda x:x[1],reverse=True))
        return pseudo_labels

    
    
    def balancing_labeled_result(self, labeled_dataset):
        label0 = []
        label1 = [] 
        for text, label in labeled_dataset:
            label = int(label)
            if label == 0:
                label0.append((text, label))
            else:
                label1.append((text, label))
        min_length = min(len(label0),len(label1))
        return label0[:min_length] + label1[:min_length]

    
    
    def gen_reverse_examples(self, labeled_examples):
        reverse_labeled_dataset = []
        for label_data in labeled_examples:
            origin_label = int(label_data.label)
            origin_text = ' '.join(label_data.text)
            
            reverse_label, reverse_text = gen_reverse_sent_with_lexicons((origin_label, origin_text), self.golden_lexicons)
            reverse_labeled_dataset.append((reverse_text ,reverse_label))
            
        reversed_examples = [torchtext.data.Example.fromlist((text, label),[('text', self.TEXT_field), ('label', self.LABEL_field)]) for (text, label) in reverse_labeled_dataset]
        return reversed_examples
        
        
        
    def train(self, num_epochs=30, dev_data=None, test_data=None):
        log = self.logger
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else -1
        model = self.model
        self.best_dev_acc = 0
        
        for epoch in range(num_epochs): 
            print('INFO current epoch: ', epoch)
            # load updated unlabeled examples dataset
            print('SSL train', len(self.unlabeled_examples))
            unlabeled_dataset = torchtext.data.Dataset(self.unlabeled_examples,
                                                       fields = [('text', self.TEXT_field), ('label', self.LABEL_field)])
            unlabel_batch_iter = torchtext.data.BucketIterator(dataset=unlabeled_dataset, batch_size=128,
                                                               sort_key=lambda x:len(x.text),  sort_within_batch=True,
                                                               device=self.device, repeat=False, shuffle=True, train=False)
            # predict unlabeled dataset
            labeled_samples = []
            
            # model evaluation
            model.eval()
            reverse_batch_iter = None
            
            with torch.no_grad():
                for idx, unlabel_batch in enumerate(unlabel_batch_iter):
                    str_list = itos(self.input_vocab, unlabel_batch.text[0])
                    str_list = [sent.replace('<pad>','').strip() for sent in str_list]
                    
                    input_var = unlabel_batch.text[0]
                    input_len = unlabel_batch.text[1]
                    target_label = unlabel_batch.label
                    
                    logits, attn = model(input_var, input_len.tolist())
                    
                    labeled_dataset = self.pseudo_labeling(input_var, str_list, logits, attn, target_label)
                    labeled_samples.extend(labeled_dataset)
            
            
            labeled_samples = list(set(labeled_samples))
            balanced_dataset = self.balancing_labeled_result(labeled_samples)
            print('INFO. # num of new balacned dataset from pseudo label result', len(balanced_dataset))
            self.save_labeled_result(self.outputs_dir+'/pseudo_label_epoch_{}.txt'.format(str(epoch)), labeled_samples)
            new_labeled_examples = self.update_dataset(labeled_samples)
            
            
            # get reversed examples from self.labeled_examples
            reversed_examples = self.gen_reverse_examples(new_labeled_examples)
            
#             with open('pseudo_label_result2.txt', 'w') as fw:
#                 for example in new_labeled_examples:
#                     text = ' '.join(example.text)
#                     label = example.label
#                     data = str(label) +'\t' + text
#                     fw.write(data +'\n')

#             with open('pseudo_label_reversed_result2.txt', 'w') as fw:
#                 for example in reversed_examples:
#                     text = ' '.join(example.text)
#                     label = example.label
#                     data = str(label) +'\t' + text
#                     fw.write(data +'\n')
            
#             1/0
            
#             reversed_dataset = torchtext.data.Dataset(reversed_examples, 
#                                                       fields = [('text', self.TEXT_field), ('label', self.LABEL_field)])
#             reversed_batch_iter = torchtext.data.BucketIterator(dataset=reversed_dataset, batch_size=128,
#                                                             sort_key=lambda x:len(x.text), sort_within_batch=True,
#                                                             device=self.device, repeat=False, shuffle=True)

            print('reverse_examples', len(reversed_examples), type(reversed_examples))
            print('self.labeled_examples', len(self.labeled_examples), type(self.labeled_examples))
        
        
        
            self.labeled_examples.extend(reversed_examples)
            
            
            #load labelled training dataset
            labeled_dataset = torchtext.data.Dataset(self.labeled_examples,
                                                     fields=[('text', self.TEXT_field), ('label', self.LABEL_field)])
            label_batch_iter = torchtext.data.BucketIterator(dataset=labeled_dataset, batch_size=128,
                                                             sort_key=lambda x:len(x.text), sort_within_batch=True,
                                                             device=self.device, repeat=False, shuffle=True)
            # Train model SSL way
            loss_total = 0
            step = 0
            model.train()
            for idx, label_batch in enumerate(label_batch_iter):
                input_variable, input_lengths = label_batch.text
                print('label', input_variable.size())
                target_variable = label_batch.label
                logits, attn = model(input_variable, input_lengths.tolist())
                self.optimizer.optimizer.zero_grad()
                loss = self.loss(logits, target_variable)
                loss.backward()
                self.optimizer.step()
                step +=1
                loss_total += loss
                logits = F.softmax(logits, dim=-1)
                prob, indice = logits.max(1)

#             for idx, reverse_batch in enumerate(reversed_batch_iter):
#                 input_variable, input_lengths = reverse_batch.text
#                 print('reverse', input_variable.size())
#                 target_variable = reverse_batch.label
#                 logits, attn = model(input_variable, input_lengths.tolist())
#                 self.optimizer.optimizer.zero_grad()
#                 loss = self.loss(logits, target_variable)
#                 loss.backward()
#                 self.optimizer.step()
#                 step +=1
#                 loss_total += loss
#                 logits = F.softmax(logits, dim=-1)
#                 prob, indice = logits.max(1)

                
            epoch_loss_avg = loss_total / step
            log_msg = "Finished epoch %d: SSL Train %s: %.4f" % (epoch, 'Cross_Entropy', epoch_loss_avg)

            if dev_data is not None:
                model.eval()
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.dev_acc = accuracy
                log_msg +=  ", Dev %s: %.4f, Accuracy: %.4f" % ('Cross_Entropy', dev_loss, accuracy)
                log.info(log_msg)

            if test_data is not None:
                model.eval()
                test_loss, accuracy = self.evaluator.evaluate(model, test_data)
                log_msg +=  ", Test %s: %.4f, Accuracy: %.4f" % ('Cross_Entropy', test_loss, accuracy)
                log.info(log_msg)
