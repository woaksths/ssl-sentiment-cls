import torch

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


def extract_lexicon(input_vocab, input_variable, target_variable, attn):
    lexicon = {0:{}, 1:{}}
    input_text = itos(input_vocab, input_variable)
    probs, indices = attn.max(-1)
    
    for idx, (attn_idx, input_var, label) in enumerate(zip(indices, input_variable, target_variable)):
        vocab_index = input_var[attn_idx.item()].item()
        word = input_vocab.itos[vocab_index]
        label = label.item()
        if label in lexicon:
            if word in lexicon[label]:
                lexicon[label][word] += 1
            else:
                lexicon[label][word] = 1
    return lexicon    


def merge_lexicon(total_dict, step_dict):
    for label in step_dict:
        for word in step_dict[label]:
            if word in total_dict[label]:
                total_dict[label][word] += step_dict[label][word] 
            else:
                total_dict[label][word] = step_dict[label][word]
    return total_dict