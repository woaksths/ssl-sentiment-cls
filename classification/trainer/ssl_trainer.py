import random
from trainer import Trainer
from evaluator import Evaluator
import os
import torchtext
import torch
import logging
from util.vocab import itos
import torch.nn.functional as F


class SSLTrainer(object):
    def __init__(self, model=None, loss=None, batch_size=64, 
                 optimizer=None, vocab=None, expt_dir=None,
                 labeled_data=None, unlabeled_data=None,
                 text_field=None, label_field=None):
        
        self._trainer= "Semi supervsed Trainer"
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
        self.vocab_dict = self.input_vocab.stoi
        
        # threshold 
        self.upper_threshold = 0.9
        self.lower_threshold = 0.5
        
        # dataset
        self.labeled_examples = labeled_data.examples # type list 
        self.unlabeled_examples = unlabeled_data.examples # type list 
        self.unlabeled_data = [example.text for example in self.unlabeled_examples]
        
        for i, example in enumerate(self.unlabeled_data):
            for j, tok in enumerate(self.unlabeled_data[i]):
                if self.vocab_dict[tok] == 0:
                    self.unlabeled_data[i][j] = '<unk>'
            self.unlabeled_data[i] = ' '.join(self.unlabeled_data[i]).strip()
        
        # field
        self.TEXT_field = text_field
        self.LABEL_field = label_field
        
        self.logger = logging.getLogger(__name__)


        
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

                
    def pseudo_labeling(self, text_list, probs, indices):
        pseudo_labels = []
        for idx, (text, label, prob) in enumerate(zip(text_list, indices, probs)):
            label = label.item()
            prob = prob.item()
            if prob > self.upper_threshold: # need to modified for novelty 
                pseudo_labels.append((text, int(label)))
        return pseudo_labels

        
    def train(self, num_epochs=30, dev_data=None):
        log = self.logger
        device = torch.device('cuda:0') if torch.cuda.is_available() else -1
        for epoch in range(num_epochs): 
            print('INFO current epoch: ', epoch)
            # load updated unlabeled examples dataset
            unlabeled_dataset = torchtext.data.Dataset(self.unlabeled_examples,
                                                       fields = [('text', self.TEXT_field)])
            unlabel_batch_iter = torchtext.data.BucketIterator(dataset=unlabeled_dataset, batch_size=8,
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
                                        
                    logits, attn = self.model(input_var, input_len.tolist())
                    logits = F.softmax(logits, dim=-1)
                    prob, indice= logits.max(1)
                    
                    labeled_data = self.pseudo_labeling(str_list, prob, indice)
                    labeled_samples.extend(labeled_data)
            
            self.save_labeled_result(self.expt_dir+'/pseudo_label_epoch{}.txt'.format(str(epoch)), labeled_samples)
            self.update_dataset(labeled_samples)
            
            # load updated labelled training dataset
            labeled_dataset = torchtext.data.Dataset(self.labeled_examples,
                                                     fields=[('text', self.TEXT_field), ('label', self.LABEL_field)])
            label_batch_iter = torchtext.data.BucketIterator(dataset=labeled_dataset, batch_size=8,
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
                logits = F.softmax(logits, dim=-1)
                
                prob, indice = logits.max(1)
                loss = self.loss(logits, target_variable)
                loss.backward()
                self.optimizer.step()
                step +=1
                loss_total += loss
            
            epoch_loss_avg = loss_total / step
            log_msg = "Finished epoch %d: SSL Train %s: %.4f" % (epoch, 'Cross_Entropy', epoch_loss_avg)
            
            if dev_data is not None:
                self.model.eval()
                dev_loss, accuracy = self.evaluator.evaluate(self.model, dev_data)
                log_msg +=  ", Dev %s: %.4f, Accuracy: %.4f" % ('Cross_Entropy', dev_loss, accuracy)
                log.info(log_msg)
