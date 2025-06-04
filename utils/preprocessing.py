import pandas as pd
import re
from utils.config import config
from collections import Counter

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"([.!?])", r" \1", text)
    text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)  # For English
    return text

def clean_hindi(text):
    text = text.strip()
    text = re.sub(r"([।.!?])", r" \1", text)
    return text

def prepare_data():
    df = pd.read_csv(config.data_path)
    df = df[['english_sentence', 'hindi_sentence']].dropna()
    
    # Clean text
    df['english'] = df['english_sentence'].apply(clean_text)
    df['hindi'] = df['hindi_sentence'].apply(clean_hindi)
    
    # Add start/end tokens to Hindi
    df['hindi'] = df['hindi'].apply(lambda x: '<start> ' + x + ' <end>')
    
    return df[['english', 'hindi']]

def build_vocab(sentences, is_hindi=False):
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence.split())
    
    vocab = {word: idx+4 for idx, (word, count) in 
             enumerate(word_counts.items()) if count >= config.min_word_count}
    
    # Add special tokens
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<end>'] = 2
    vocab['<unk>'] = 3
    
    return vocab