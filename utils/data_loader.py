import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.preprocessing import prepare_data, build_vocab

class TranslationDataset(Dataset):
    def __init__(self, english_sentences, hindi_sentences, eng_vocab, hin_vocab):
        self.english_sentences = english_sentences
        self.hindi_sentences = hindi_sentences
        self.eng_vocab = eng_vocab
        self.hin_vocab = hin_vocab
        
    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, idx):
        eng_sentence = self.english_sentences[idx]
        hin_sentence = self.hindi_sentences[idx]
        
        eng_ids = [self.eng_vocab.get(word, self.eng_vocab['<unk>']) 
                  for word in eng_sentence.split()]
        hin_ids = [self.hin_vocab.get(word, self.hin_vocab['<unk>']) 
                  for word in hin_sentence.split()]
        
        return {
            'english': torch.tensor(eng_ids, dtype=torch.long),
            'hindi': torch.tensor(hin_ids, dtype=torch.long)
        }

def collate_fn(batch):
    eng_batch = [item['english'] for item in batch]
    hin_batch = [item['hindi'] for item in batch]
    
    eng_padded = torch.nn.utils.rnn.pad_sequence(
        eng_batch, padding_value=0, batch_first=True)
    hin_padded = torch.nn.utils.rnn.pad_sequence(
        hin_batch, padding_value=0, batch_first=True)
    
    return eng_padded, hin_padded

def get_data_loaders():
    df = prepare_data()
    eng_sentences = df['english'].tolist()
    hin_sentences = df['hindi'].tolist()
    
    # Split data
    split_idx = int(len(eng_sentences) * config.train_ratio)
    train_eng = eng_sentences[:split_idx]
    train_hin = hin_sentences[:split_idx]
    val_eng = eng_sentences[split_idx:]
    val_hin = hin_sentences[split_idx:]
    
    # Build vocabularies
    eng_vocab = build_vocab(train_eng)
    hin_vocab = build_vocab(train_hin, is_hindi=True)
    
    # Create datasets
    train_dataset = TranslationDataset(train_eng, train_hin, eng_vocab, hin_vocab)
    val_dataset = TranslationDataset(val_eng, val_hin, eng_vocab, hin_vocab)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, 
        shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, 
        shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, eng_vocab, hin_vocab