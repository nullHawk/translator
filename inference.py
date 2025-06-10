import torch
from utils.config import config
from utils.preprocessing import clean_text, clean_hindi
from utils.data_loader import TranslationDataset
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq
import pickle

def translate_sentence(sentence, model, eng_vocab, hin_vocab, device):
    model.eval()
    sentence = clean_text(sentence)
    
    # Convert to tensor
    tokens = [eng_vocab.get(word, eng_vocab['<unk>']) for word in sentence.split()]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    
    trg_indexes = [hin_vocab['<start>']]
    
    for _ in range(config.max_length):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
        
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, encoder_outputs)
        
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        if pred_token == hin_vocab['<end>']:
            break
    
    trg_tokens = [list(hin_vocab.keys())[list(hin_vocab.values()).index(i)] 
                 for i in trg_indexes]
    
    return ' '.join(trg_tokens[1:-1])  # Remove <start> and <end>

def main():
    # Load vocabularies
    with open('bin/eng_vocab.pkl', 'rb') as f:
        eng_vocab = pickle.load(f)
    with open('bin/hin_vocab.pkl', 'rb') as f:
        hin_vocab = pickle.load(f)
    
    # Load model
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
    model.load_state_dict(torch.load("bin/seq2seq.pth", map_location=config.device))
    
    # Interactive translation
    while True:
        sentence = input("Enter English sentence (type 'exit' to quit): ")
        if sentence.lower() == 'exit':
            break
        translation = translate_sentence(sentence, model, eng_vocab, hin_vocab, config.device)
        print(f"Hindi Translation: {translation}\n")

if __name__ == "__main__":
    main()