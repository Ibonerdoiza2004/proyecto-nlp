import os
import sys
import math
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle

class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_unificado.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'generacion_texto')
    MODEL_PATH = os.path.join(MODEL_DIR, 'enc_dec_trans_scratch.pt')
    VOCAB_PATH = os.path.join(MODEL_DIR, 'enc_dec_trans_scratch_vocab.pkl')
    
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1
    
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 0.0005
    MAX_LEN = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.w2i = {}
        self.i2w = {}
        self.size = 0
        
    def build(self, texts):
        c = Counter()
        for t in texts: c.update(t.lower().split())
        specials = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.w2i = {w:i for i,w in enumerate(specials)}
        idx = len(specials)
        for w, count in c.items():
            if count >= self.min_freq:
                self.w2i[w] = idx
                idx += 1
        self.i2w = {i:w for w,i in self.w2i.items()}
        self.size = len(self.w2i)
        
    def encode(self, text):
        return [2] + [self.w2i.get(w, 1) for w in text.lower().split()] + [3]
    
    def decode(self, idxs):
        return ' '.join([self.i2w.get(i, '<UNK>') for i in idxs if i not in [0, 2, 3]])

class TextDataset(Dataset):
    def __init__(self, texts, vocab):
        self.vocab = vocab
        self.pairs = []
        for i in range(len(texts)-1):
            if len(texts[i].split()) > 5:
                self.pairs.append((texts[i], texts[i+1]))
                
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_idx = self.vocab.encode(src)[:Config.MAX_LEN]
        tgt_idx = self.vocab.encode(tgt)[:Config.MAX_LEN]
        src_idx += [0]*(Config.MAX_LEN - len(src_idx))
        tgt_idx += [0]*(Config.MAX_LEN - len(tgt_idx))
        return torch.tensor(src_idx), torch.tensor(tgt_idx)

class Transformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, Config.D_MODEL)
        self.pos = self._get_pos_encoding()
        
        self.transformer = nn.Transformer(
            d_model=Config.D_MODEL,
            nhead=Config.N_HEADS,
            num_encoder_layers=Config.N_LAYERS,
            num_decoder_layers=Config.N_LAYERS,
            dim_feedforward=Config.D_FF,
            dropout=Config.DROPOUT,
            batch_first=True
        )
        if hasattr(self.transformer.encoder, 'enable_nested_tensor'):
            self.transformer.encoder.enable_nested_tensor = False
        if hasattr(self.transformer.encoder, 'use_nested_tensor'):
            self.transformer.encoder.use_nested_tensor = False
            
        self.fc = nn.Linear(Config.D_MODEL, vocab_size)
    
    def _get_pos_encoding(self):
        pe = torch.zeros(Config.MAX_LEN, Config.D_MODEL)
        pos = torch.arange(0, Config.MAX_LEN).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, Config.D_MODEL, 2).float() * (-math.log(10000.0) / Config.D_MODEL))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return nn.Parameter(pe.unsqueeze(0), requires_grad=False)
        
    def forward(self, src, tgt):
        src_emb = self.emb(src) + self.pos[:, :src.size(1), :]
        tgt_emb = self.emb(tgt) + self.pos[:, :tgt.size(1), :]
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device).bool()
        
        src_padding = (src == 0)
        tgt_padding = (tgt == 0)
        
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, 
                               src_key_padding_mask=src_padding,
                               tgt_key_padding_mask=tgt_padding,
                               memory_key_padding_mask=src_padding)
        return self.fc(out)

def train():
    df = pd.read_csv(Config.DATASET_PATH)
    texts = df[df['speaker'] == 'MIGUEL']['text'].tolist()
    
    vocab = Vocabulary()
    vocab.build(texts)
    with open(Config.VOCAB_PATH, 'wb') as f: pickle.dump(vocab, f)
    
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    
    train_dl = DataLoader(TextDataset(train_texts, vocab), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(TextDataset(val_texts, vocab), batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = Transformer(vocab.size).to(Config.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for src, tgt in train_dl:
            src, tgt = src.to(Config.DEVICE), tgt.to(Config.DEVICE)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            
            opt.zero_grad()
            out = model(src, tgt_in)
            loss = crit(out.reshape(-1, vocab.size), tgt_out.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(Config.DEVICE), tgt.to(Config.DEVICE)
                tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
                out = model(src, tgt_in)
                val_loss += crit(out.reshape(-1, vocab.size), tgt_out.reshape(-1)).item()
        
        print(f"Epoch {epoch+1}: Train Loss {total_loss/len(train_dl):.4f} | Val Loss {val_loss/len(val_dl):.4f}")
        torch.save(model.state_dict(), Config.MODEL_PATH)

def generate(text):
    with open(Config.VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    model = Transformer(vocab.size).to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()
    
    # Parámetros de generación
    temperature = 0.7
    top_k = 40
    repetition_penalty = 1.2
    
    src = torch.tensor(vocab.encode(text)[:Config.MAX_LEN] + [0]*Config.MAX_LEN).unsqueeze(0).to(Config.DEVICE)[:, :Config.MAX_LEN]
    tgt = [[2]]
    
    for _ in range(Config.MAX_LEN):
        tgt_tensor = torch.tensor(tgt).to(Config.DEVICE)
        with torch.no_grad():
            out = model(src, tgt_tensor)
        
        # Obtener logits del último token generado y aplicar temperatura
        logits = out[0, -1] / temperature
        
        # Penalización de repetición
        for token_id in set(tgt[0]):
            if token_id in [0, 1, 2, 3]: continue
            logits[token_id] /= repetition_penalty
            
        # Evitar <UNK> y <PAD>
        logits[1] = -1e9
        logits[0] = -1e9
        
        # Filtrado Top-K
        v, idx = torch.topk(logits, top_k)
        probs = torch.softmax(v, dim=-1)
        next_tok = idx[torch.multinomial(probs, 1)].item()
        
        if next_tok == 3: break
        tgt[0].append(next_tok)
        
    print("Generado:", vocab.decode(tgt[0]))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--train': train()
    elif len(sys.argv) > 1 and sys.argv[1] == '--demo':
        while True:
            t = input("Tú: ")
            if t == 'salir': break
            generate(t)
    else: print("Use --train or --demo")