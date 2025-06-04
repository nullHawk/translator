import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.config import config
from utils.data_loader import get_data_loaders
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)

def train():
    train_loader, val_loader, eng_vocab, hin_vocab = get_data_loaders()
    
    # Model initialization
    enc = Encoder(
        len(eng_vocab), 
        config.embedding_dim, 
        config.hidden_size, 
        config.num_layers, 
        config.dropout
    ).to(config.device)
    
    dec = Decoder(
        len(hin_vocab),
        config.embedding_dim,
        config.hidden_size,
        config.num_layers,
        config.dropout
    ).to(config.device)
    
    model = Seq2Seq(enc, dec, config.device).to(config.device)
    model.apply(init_weights)
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            src, trg = src.to(config.device), trg.to(config.device)
            
            optimizer.zero_grad()
            output = model(src, trg, config.teacher_forcing_ratio)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch: {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Save model
        torch.save(model.state_dict(), f"seq2seq_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train()