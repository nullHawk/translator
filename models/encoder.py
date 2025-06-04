import torch.nn as nn
from utils.config import config

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded: [batch_size, src_len, embedding_dim]
        
        outputs, hidden = self.rnn(embedded.permute(1, 0, 2))
        # outputs: [src_len, batch_size, hidden_dim]
        # hidden: [n_layers * num_directions, batch_size, hidden_dim]
        
        return outputs, hidden