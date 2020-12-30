import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, embedding, max_len, hidden_size, input_dropout_p=0, dropout_p=0,
                n_layers=1, bidirectional=False, variable_lengths=False, update_embedding=True):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
#         self.embedding = nn.Embedding(vocab_size, 300)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.variable_lengths= variable_lengths
        self.embedding.weight.requires_grad = True

        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers= n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=0)
        self.dropout_p = dropout_p
        
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size, n_layers, 
                            batch_first=True, bidirectional=bidirectional, dropout=0)
        if bidirectional is True:
            self.classifier = nn.Linear(self.hidden_size*2, 2) # 2 -> clsas_label 
        else:
            self.classifier = nn.Linear(self.hidden_size, 2)
        
        self.mask = None

        
    def attention(self, outputs, states):
        # states -> 1, batch, hidden
        hidden = states.squeeze(0) # batch, hidden
        attn = torch.bmm(outputs, hidden.unsqueeze(2)).squeeze(2) # batch, seq_len
        if self.mask is not None:
            attn.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn, 1)
        new_hidden_state = torch.bmm(outputs.transpose(1,2), attn.unsqueeze(2)).squeeze(2)
        return new_hidden_state, attn

    
    def get_mask(self, input_var, masking_indexs):
        self.mask = torch.eq(input_var, 1) # 1 is pad index
        for idx in masking_indexs:
            self.mask = self.mask | torch.eq(input_var, idx) 
            
            
    def forward(self, input_var, input_lengths, masking_indexs=None):
#         self.get_mask(input_var, masking_indexs)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, (final_hidden_state, final_cell_state) = self.lstm(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        c_t, attn = self.attention(output, final_hidden_state)
        logits = self.classifier(c_t)
#         logits = F.tanh(logits)
        return logits, attn


class Classifier_Attention_LSTM(nn.Module):
    def __init__(self, embedding, n_labels):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
            self.embedding.weight.requires_grad = True

        self.rnn = nn.LSTM(300, 300, batch_first=True)
        self.tanh1 = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(300))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(300,n_labels)

    def forward(self,token2, seq_lengths):
        embedded = self.embedding(token2)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, seq_lengths, batch_first=True)
        packed_output,(final_hidden_state, final_cell_state) = self.rnn(packed_input)
        r_output, input_sizes =  nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = r_output
        #output = self.tanh1(r_output)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        #print("##torch.matmul(output, self.w)##:", torch.matmul(output, self.w))
        alpha = F.softmax(torch.matmul(output, self.w)).unsqueeze(-1)  # [128, 32, 1]
        out = r_output * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        return out, alpha.squeeze()
    
    