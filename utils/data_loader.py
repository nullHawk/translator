import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.preprocessing import prepare_data, build_vocab
from utils.config import config

class TranslationDataset(Dataset):
    def __init__(self, english_sentences, hindi_sentences, eng_vocab, hin_vocab):
        self.english_sentences = english_sentences
        self.hindi_sentences = hindi_sentences
        self.eng_vocab = eng_vocab
        self.hin_vocab = hin_vocab
        self.eng_vocab_size = len(eng_vocab)
        self.hin_vocab_size = len(hin_vocab)
        
    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self, idx):
        eng_sentence = self.english_sentences[idx]
        hin_sentence = self.hindi_sentences[idx]
        
        eng_ids = [self.eng_vocab.get(word, self.eng_vocab['<unk>']) 
                  for word in eng_sentence.split()]
        hin_ids = [self.hin_vocab.get(word, self.hin_vocab['<unk>']) 
                  for word in hin_sentence.split()]
        
        # Clamp indices to vocabulary size
        eng_ids = [min(idx, self.eng_vocab_size - 1) for idx in eng_ids]
        hin_ids = [min(idx, self.hin_vocab_size - 1) for idx in hin_ids]
        
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
    df = df.sample(frac=0.1, random_state=42)
    df['eng_len'] = df['english'].apply(lambda x: len(x.split()))
    df['hin_len'] = df['hindi'].apply(lambda x: len(x.split()))
    df = df[(df['eng_len'] <= config.max_length) & 
            (df['hin_len'] <= config.max_length)]
    
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

    # Save vocabularies for inference
    with open('eng_vocab.pkl', 'wb') as f:
        pickle.dump(eng_vocab, f)
    with open('hin_vocab.pkl', 'wb') as f:
        pickle.dump(hin_vocab, f)
    print(f"English vocabulary size: {len(eng_vocab)}")
    print(f"Hindi vocabulary size: {len(hin_vocab)}")
    print(f"Max English index: {max(eng_vocab.values())}")
    print(f"Max Hindi index: {max(hin_vocab.values())}")
    
    return train_loader, val_loader, eng_vocab, hin_vocab