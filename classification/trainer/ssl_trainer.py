import random
from trainer import Trainer
from evaluator import Evaluator
import os
import torchtext
import torch
import logging
from util.vocab import itos
import torch.nn.functional as F
from util.augment import *

class SSLTrainer(object):
    def __init__(self, model=None, loss=None, batch_size=64, 
                 optimizer=None, vocab=None, expt_dir=None,
                 golden_lexicons=None, labeled_data=None, unlabeled_data=None,
                 text_field=None, label_field=None):
        
        self._trainer= "Semi supervised Trainer"
        self.model = model
        self.loss = loss
        self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
        self.optimizer = optimizer

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        
        self.batch_size = batch_size
        self.input_vocab = vocab
        
        # threshold 
        self.upper_threshold = 0.9
        self.lower_threshold = 0.5

        self.golden_lexicons = golden_lexicons
        print(self.golden_lexicons)
        self.candidate_lexicons = self.augment_lexicons(golden_lexicons)
        print(self.candidate_lexicons)
        
        print('labeled_data', len(labeled_data.examples))
        print('unlabeled_data' ,len(unlabeled_data.examples))        
        
        # dataset
        self.labeled_examples = labeled_data.examples # type list 
        self.unlabeled_examples = unlabeled_data.examples # type list       
        
        
#         self.unlabeled_data = [example.text for example in self.unlabeled_examples]
#         for i, example in enumerate(self.unlabeled_data):
#             for j, tok in enumerate(self.unlabeled_data[i]):
#                 if self.input_vocab.stoi[tok] == 0:
#                     self.unlabeled_data[i][j] = '<unk>'
#             self.unlabeled_data[i] = ' '.join(self.unlabeled_data[i]).strip()
        
        
        # field
        self.TEXT_field = text_field
        self.LABEL_field = label_field
        self.logger = logging.getLogger(__name__)
        
        self.match = 0
        self.total = 0
        
        
    def augment_lexicons(self, golden_lexicons):
        # get candidate lexicons from wordnet
        candidate = {}
        for label in golden_lexicons:
            label = int(label)
            if label not in candidate:
                candidate[label] = set()
            for word in golden_lexicons[label]:
                synonyms = get_synonyms(word, '')
                for synonym in synonyms:
                    candidate[label].add(synonym)
        return candidate
                
        
    def update_dataset(self, labeled_dataset):
        print('*'*100)
        print('INFO. # num of current training dataset', len(self.labeled_examples))
        print('INFO. # num of current unlabeled dataset', len(self.unlabeled_data), len(self.unlabeled_examples))
        print('INFO. # num of relabeled dataset', len(labeled_dataset))

        for idx, (text, label) in  enumerate(labeled_dataset):
            text = text.strip()
            # remove labeled text from unlabeled_text
            if text in self.unlabeled_data:
                self.unlabeled_data.remove(text)

            # append labeled text into labeled_set    
            example = torchtext.data.Example.fromlist((text, label),
                                                      [('text', self.TEXT_field),
                                                       ('label', self.LABEL_field)])
            self.labeled_examples.append(example)

        # update unlabeled examples         
        unlabeled_examples = [torchtext.data.Example.fromlist([text],[('text', self.TEXT_field)]) \
                              for text in self.unlabeled_data]
        self.unlabeled_examples = unlabeled_examples
        print('INFO. # num of remain unlabeled data', len(self.unlabeled_data), len(self.unlabeled_examples))
        print('INFO. # updated labeled data ', len(self.labeled_examples))
        

    def save_labeled_result(self, path, labeled_result):
        with open(path, 'w') as fw:
            for text, label in labeled_result:
                fw.write(text +'\t' + str(label) +'\n')

    
    def match_by_lexicon(self, text, golden_lexicons):
        #### 추후에 .... match 할 때, word가 golden_lexicons에 포함되는데 이전 단어에 not이 붙어있다면, 반대로 레이블을 지정.
        ### 반경 2-window 이내 negator가 있는 경우 해당 클래스의 lexicon이라 하지 않고, 반대 렉시콘이라고 지정. 
        match_count = {label:0 for label in golden_lexicons.keys()}
        matched_tokens = []
        for word in text.split(' '):
            for label in golden_lexicons:
                if word in golden_lexicons[label]:
                    match_count[label] += 1
                    matched_tokens.append(word)
        return match_count, matched_tokens
    
    
    def pseudo_labeling(self, text_list, predicted_labels,  confidences, attns, targets):
        print('pseudo label')
        pseudo_labels = []
        for idx, (text, label, prob, attn, target) in enumerate(zip(text_list, predicted_labels, confidences, attns,targets)):
            pred = label.item()
            prob = prob.item()
            matching_result, matched_tokens = self.match_by_lexicon(text, self.golden_lexicons)
            pred_label = None
            
            ## Threshold 조절
            if matching_result[0] > matching_result[1] and  matching_result[0] >= 2:
                pred_label = 0
            elif matching_result[0] < matching_result[1] and  matching_result[1] >= 2:
                pred_label = 1
            if pred_label is not None and prob > 0.6:
                self.total +=1
                # 추후에 error analysis 진행...!!!!!
                print('*'*100)
                print(text)
                print(matching_result)
                print('lexicon pred:', pred_label, ', target:', target.item())
                print('model pred:', pred, 'confidence:', prob)
                print('matched_tokens', matched_tokens)
                if pred_label == target.item():
                    self.match +=1
                print(self.total, self.match)
        return pseudo_labels

    
    def train(self, num_epochs=30, dev_data=None, test_data=None):
        log = self.logger
        device = torch.device('cuda:0') if torch.cuda.is_available() else -1
        for epoch in range(num_epochs): 
            print('INFO current epoch: ', epoch)
            # load updated unlabeled examples dataset
            print('SSL train', len(self.unlabeled_examples))
            unlabeled_dataset = torchtext.data.Dataset(self.unlabeled_examples,
                                                       fields = [('text', self.TEXT_field), ('label', self.LABEL_field)])
            unlabel_batch_iter = torchtext.data.BucketIterator(dataset=unlabeled_dataset, batch_size=1024,
                                                               sort_key=lambda x:len(x.text),  sort_within_batch=True,
                                                               device=device, repeat=False, shuffle=True, train=False)
            # predict unlabeled dataset
            labeled_samples = []
            # model evaluation
            self.model.eval()
            
            with torch.no_grad():
                for idx, unlabel_batch in enumerate(unlabel_batch_iter):
                    str_list = itos(self.input_vocab, unlabel_batch.text[0])
                    str_list = [sent.replace('<pad>','').strip() for sent in str_list]
                    
                    input_var = unlabel_batch.text[0]
                    input_len = unlabel_batch.text[1]
                    target_label = unlabel_batch.label
                    
                    logits, attn = self.model(input_var, input_len.tolist())
                    logits = F.softmax(logits, dim=-1)
                    confidence, indice= logits.max(1)
                    
                    self.pseudo_labeling(str_list, indice, confidence, attn, target_label)
#                     labeled_samples.extend(labeled_data)
                    1/0
            self.save_labeled_result(self.expt_dir+'/pseudo_label_epoch{}.txt'.format(str(epoch)), labeled_samples)
            self.update_dataset(labeled_samples)
            
            # load updated labelled training dataset
            labeled_dataset = torchtext.data.Dataset(self.labeled_examples,
                                                     fields=[('text', self.TEXT_field), ('label', self.LABEL_field)])
            label_batch_iter = torchtext.data.BucketIterator(dataset=labeled_dataset, batch_size=256,
                                                             sort_key=lambda x:len(x.text), sort_within_batch=True,
                                                             device=device, repeat=False, shuffle=True)
            
            # Train model SSL way
            self.model.train()
            loss_total = 0
            step = 0
            
            for idx, label_batch in enumerate(label_batch_iter):
                input_variable, input_lengths = label_batch.text
                target_variable = label_batch.label
                self.model.zero_grad()
                logits, attn = self.model(input_variable, input_lengths.tolist())
                loss = self.loss(logits, target_variable)
                loss.backward()
                self.optimizer.step()
                self.optimizer.optimizer.zero_grad()
                step +=1
                loss_total += loss
                logits = F.softmax(logits, dim=-1)
                prob, indice = logits.max(1)
                
            epoch_loss_avg = loss_total / step
            log_msg = "Finished epoch %d: SSL Train %s: %.4f" % (epoch, 'Cross_Entropy', epoch_loss_avg)
            
            if dev_data is not None:
                self.model.eval()
                dev_loss, accuracy = self.evaluator.evaluate(self.model, dev_data)
                log_msg +=  ", Dev %s: %.4f, Accuracy: %.4f" % ('Cross_Entropy', dev_loss, accuracy)
                log.info(log_msg)

            if test_data is not None:
                self.model.eval()
                test_loss, accuracy = self.evaluator.evaluate(self.model, test_data)
                log_msg +=  ", Test %s: %.4f, Accuracy: %.4f" % ('Cross_Entropy', test_loss, accuracy)
                log.info(log_msg)
