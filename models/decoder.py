import torch.nn as nn
import torch
from models.attention import Attention
from utils.config import config

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.attention = Attention(hidden_dim)
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim + hidden_dim, 
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
        # input: [batch_size]
        # hidden: [n_layers, batch_size, hidden_dim]
        # encoder_outputs: [src_len, batch_size, hidden_dim]
        
        input = input.unsqueeze(0)
        # input: [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded: [1, batch_size, embedding_dim]
        
        a = self.attention(hidden[-1], encoder_outputs)
        # a: [src_len, batch_size]
        
        a = a.permute(1, 0).unsqueeze(1)
        # a: [batch_size, 1, src_len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        
        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        # weighted: [1, batch_size, hidden_dim]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input: [1, batch_size, embedding_dim + hidden_dim]
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [1, batch_size, hidden_dim]
        # hidden: [n_layers, batch_size, hidden_dim]
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        prediction = self.fc_out(torch.cat((output, weighted), dim=1))
        # prediction: [batch_size, output_dim]
        
        return prediction, hidden