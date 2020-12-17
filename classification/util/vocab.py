import torch
import torch.nn.functional as F
from util.lexicon_config import STOP_WORDS, END_WORDS 

def itos(vocab, input_var):
    # input_var -> # batch, seq_len
    # vocab -> GloVe vector
    assert input_var.dim() == 2
    result = []
    for batch in input_var:
        res = []
        for idx in batch:
            res.append(vocab.itos[idx])
        result.append(' '.join(res))
    return result


def itos_sent(vocab, input_var):
    # input_var -> # batch, seq_len
    # vocab -> GloVe vector
    assert input_var.dim() == 1
    result = []
    for idx in input_var:
        result.append(vocab.itos[idx])
    return ' '.join(result)


def extract_lexicon(input_vocab, input_variable, target_variable, logits, attn):
    lexicon = {0:[], 1:[]}
    attn_probs, attn_indices = attn.topk(k=10, dim=-1)
    logits = F.softmax(logits, dim=-1)
    confidences, indices = logits.max(dim=-1)
    for idx, (confidence, (pred, label)) in enumerate(zip(confidences, zip(indices, target_variable))):
        confidence = confidence.item()
        pred = pred.item()
        label = label.item()
        if confidence >= 0.9 and pred == label:
            lexicon_idx = [input_variable[idx][top_idx] for top_idx in attn_indices[idx]]
            word_set = []
            for vocab_idx in lexicon_idx:
                word = input_vocab.itos[vocab_idx.item()] 
                if word in STOP_WORDS or word in END_WORDS:
                    continue
                else:
                    word_set.append(word)
            lexicon[label].append(word_set)        
#             word_set = [input_vocab.itos[vocab_idx.item()] for vocab_idx in lexicon_idx]
#             lexicon[label].append(word_set)
#             print('label: {}'.format(label), word_set)
    return lexicon


def merge_lexicon(total_dict, step_dict):
    for label in step_dict:
        for word in step_dict[label]:
            if word in total_dict[label]:
                total_dict[label][word] += step_dict[label][word] 
            else:
                total_dict[label][word] = step_dict[label][word]
    return total_dict


