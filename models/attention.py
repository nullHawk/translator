import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: [1, batch_size, hidden_dim]
        # encoder_outputs: [src_len, batch_size, hidden_dim]

        src_len = encoder_outputs.shape[0]
        hidden = hidden.repeat(src_len, 1, 1)
        # hidden: [src_len, batch_size, hidden_dim]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [src_len, batch_size, hidden_dim]
        
        attention = self.v(energy).squeeze(2)
        # attention: [src_len, batch_size]
        
        return F.softmax(attention, dim=0)