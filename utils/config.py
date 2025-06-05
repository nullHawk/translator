import torch

class Config:
    # Data
    data_path = "data/hindi_english_parallel.csv"
    train_ratio = 0.8
    
    # Preprocessing
    max_length = 20
    min_word_count = 3
    
    # Model
    embedding_dim = 256
    hidden_size = 512
    num_layers = 2
    dropout = 0.5
    
    # Training
    batch_size = 64
    learning_rate = 0.001
    epochs = 20
    teacher_forcing_ratio = 0.5

    max_vocab_english = 20000
    max_vocab_hindi = 50000
    max_length = 20  # Maximum sentence length
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
config = Config()