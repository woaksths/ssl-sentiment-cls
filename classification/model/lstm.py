import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, embedding, max_len, hidden_size, input_dropout_p=0, dropout_p=0,
                n_layers=1, bidirectional=False, variable_lengths=False, update_embedding=True):
        
        super(LSTM, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding)
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        
        self.variable_lengths= variable_lengths
        
        # embedding
        self.embedding.weight.requires_grad = update_embedding
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers= n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.dropout_p = dropout_p
        
        self.lstm = nn.LSTM(self.embedding.embedding_dim, hidden_size, n_layers, 
                            batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.classifier = nn.Linear(self.hidden_size*2, 2) # 2 -> clsas_label 
    
    
    def attention(self, outputs, states):
        # outputs -> (#batch_size, #seq_len, #hidden_dim*2)
        merged_state = torch.cat([s for s in states], dim=-1) 
        merged_state = merged_state.unsqueeze(2) # (batch_size, hidden_dim*2, 1)
        
        attn = torch.bmm(outputs, merged_state) #batch_size, #seq_len, # 1

        # NEED ATTENTION MASKING
        attn = torch.nn.functional.softmax(attn.squeeze(2), dim=-1).unsqueeze(2) #batch_size, # seq_len, #1
        c_t = torch.bmm(torch.transpose(outputs, 1,2), attn).squeeze(2) # batch_size, hidden_dim*2
        return c_t, attn
    
        
    def forward(self, input_var, input_lengths):        
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