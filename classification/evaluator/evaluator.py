from __future__ import print_function, division

import torch
import torchtext

from model import LSTM
import torch.nn.functional as F


class Evaluator(object):
    
    """ Class to evaluate models with given datasets.
    Args:
        loss: loss for evaluator (default: cross entropy loss)
        batch_size (int, optional): batch size for evaluator 
    """

    def __init__(self, loss=None, batch_size=64):
        self.loss = loss
        self.batch_size = batch_size

    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model: model to evaluate
            data : dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()
        loss = self.loss
#         loss.reset()
        match = 0
        total = 0

        device = torch.device('cuda:0') if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.text),
            device=device, train=False, repeat=False)
        
        acc_loss = 0
        idx = 0
        with torch.no_grad():
            for batch in batch_iterator:
                input_variables, input_lengths = getattr(batch, 'text')
                target_variables = getattr(batch, 'label')
                
                logits, attn = model(input_variables, input_lengths.tolist())
                # Evaluation
                logits = F.softmax(logits, dim=-1)
                prob, indice= logits.max(1)
                
                loss_per = loss(logits, target_variables)
                acc_loss += loss_per
                                
                correct = target_variables.eq(indice).sum().item()
                match += correct
                total += logits.size()[0]
                idx += 1
                
        if total == 0:
            accuracy = float('nan')
        else:            
            accuracy = match / total
            loss = acc_loss / idx
        return loss, accuracy