from utils.config import config
from inference import main, translate_sentence
from models.encoder import Encoder
from models.decoder import Decoder
from models.seq2seq import Seq2Seq

import gradio as gr
import pickle
import torch

def translate(inp):
    global model, eng_vocab, hin_vocab
    text = translate_sentence(inp, model, eng_vocab, hin_vocab, config.device)
    return text

def main():
    global model, eng_vocab, hin_vocab
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
    

    app = gr.Interface(
        fn=translate,
        inputs='textbox',
        outputs='textbox'
    )

    app.launch()


if __name__ == "__main__":
    main()