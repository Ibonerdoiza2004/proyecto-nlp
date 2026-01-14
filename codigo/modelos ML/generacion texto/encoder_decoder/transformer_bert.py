import os
import sys
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import pickle


class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_unificado.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'generacion_texto')
    MODEL_PATH = os.path.join(MODEL_DIR, 'enc_dec_trans_bert.pt')
    VOCAB_PATH = os.path.join(MODEL_DIR, 'enc_dec_trans_bert_vocab.pkl')
    
    BETO_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 3
    D_FF = 1024
    DROPOUT = 0.1
    
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0005
    MAX_INPUT_LEN = 64
    MAX_OUTPUT_LEN = 100
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# VOCABULARIO
class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            word_counts.update(text.lower().split())
        
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        idx = len(special_tokens)
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def encode(self, text, max_len=None):
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in text.lower().split()]
        if max_len: indices = indices[:max_len-2]
        indices = [self.word2idx['<SOS>']] + indices + [self.word2idx['<EOS>']]
        return indices
    
    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices if idx not in [0, 2, 3]])
    
    def save(self, path):
        with open(path, 'wb') as f: pickle.dump(self.__dict__, f)
    
    def load(self, path):
        with open(path, 'rb') as f: self.__dict__.update(pickle.load(f))


# DATASET
class Dataset(Dataset):
    def __init__(self, texts, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.pairs = []
        for i in range(len(texts)-1):
            if len(texts[i].split()) > 5:
                self.pairs.append((texts[i], texts[i+1]))
    
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        inp, out = self.pairs[idx]
        tokens = self.tokenizer(inp, max_length=Config.MAX_INPUT_LEN, padding='max_length', truncation=True, return_tensors='pt')
        tgt = self.vocab.encode(out, Config.MAX_OUTPUT_LEN)
        tgt += [0] * (Config.MAX_OUTPUT_LEN - len(tgt))
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0), torch.tensor(tgt[:Config.MAX_OUTPUT_LEN])


# MODELO
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x): return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.encoder_bert = AutoModel.from_pretrained(Config.BETO_MODEL, add_pooling_layer=False)
        for p in self.encoder_bert.parameters(): p.requires_grad = False
        for p in self.encoder_bert.encoder.layer[-2:].parameters(): p.requires_grad = True
        
        self.proj = nn.Linear(768, Config.D_MODEL)
        self.embedding = nn.Embedding(vocab_size, Config.D_MODEL)
        self.pos_encoder = PositionalEncoding(Config.D_MODEL)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=Config.D_MODEL, nhead=Config.N_HEADS, dim_feedforward=Config.D_FF, dropout=Config.DROPOUT, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=Config.N_LAYERS)
        self.fc_out = nn.Linear(Config.D_MODEL, vocab_size)
    
    def forward(self, src_ids, src_mask, tgt):
        bert_out = self.encoder_bert(input_ids=src_ids, attention_mask=src_mask).last_hidden_state
        memory = self.proj(bert_out)
        
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(Config.D_MODEL))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device).bool()
        
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=(src_mask==0))
        return self.fc_out(output)
    
    def unfreeze_all(self):
        for param in self.encoder_bert.parameters():
            param.requires_grad = True
        print("BERT completamente descongelado para fine-tuning")

# MAIN
def train():
    print("ENCODER-DECODER: BETO + TRANSFORMER DECODER")
    
    df = pd.read_csv(Config.DATASET_PATH)
    texts = df[df['speaker'] == 'MIGUEL']['text'].tolist()
    
    vocab = Vocabulary()
    vocab.build_vocab(texts)
    vocab.save(Config.VOCAB_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.BETO_MODEL)
    
    # Split datos
    split = int(len(texts) * 0.9)
    train_dl = DataLoader(Dataset(texts[:split], vocab, tokenizer), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(Dataset(texts[split:], vocab, tokenizer), batch_size=Config.BATCH_SIZE)
    
    model = TransformerModel(vocab.vocab_size).to(Config.DEVICE)
    crit = nn.CrossEntropyLoss(ignore_index=0)
    
    # FASE 1: BERT CONGELADO
    print("FASE 1: Entrenamiento con BERT parcialmente congelado")
    
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
    best_loss, patience = float('inf'), 0
    PATIENCE = 5
    
    for epoch in range(Config.EPOCHS):
        model.train()
        total_loss = 0
        for src, mask, tgt in train_dl:
            src, mask, tgt = src.to(Config.DEVICE), mask.to(Config.DEVICE), tgt.to(Config.DEVICE)
            opt.zero_grad()
            out = model(src, mask, tgt[:, :-1])
            loss = crit(out.reshape(-1, vocab.vocab_size), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            total_loss += loss.item()
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, mask, tgt in val_dl:
                src, mask, tgt = src.to(Config.DEVICE), mask.to(Config.DEVICE), tgt.to(Config.DEVICE)
                out = model(src, mask, tgt[:, :-1])
                val_loss += crit(out.reshape(-1, vocab.vocab_size), tgt[:, 1:].reshape(-1)).item()
        val_loss /= len(val_dl)
        
        print(f"[FASE 1] Epoch {epoch+1}: Train {total_loss/len(train_dl):.4f} | Val {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  Guardado (val_loss: {best_loss:.4f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping en epoch {epoch+1}")
                break
    
    # FASE 2: FINE-TUNING COMPLETO
    print("FASE 2: Fine-tuning completo (BERT descongelado)")
    
    model.unfreeze_all()
    opt_ft = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ft, T_max=20)
    
    best_loss_ft, patience = best_loss, 0
    
    for epoch in range(20):
        model.train()
        total_loss = 0
        for src, mask, tgt in train_dl:
            src, mask, tgt = src.to(Config.DEVICE), mask.to(Config.DEVICE), tgt.to(Config.DEVICE)
            opt_ft.zero_grad()
            out = model(src, mask, tgt[:, :-1])
            loss = crit(out.reshape(-1, vocab.vocab_size), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt_ft.step()
            total_loss += loss.item()
        
        scheduler_ft.step()
        
        # Validación
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, mask, tgt in val_dl:
                src, mask, tgt = src.to(Config.DEVICE), mask.to(Config.DEVICE), tgt.to(Config.DEVICE)
                out = model(src, mask, tgt[:, :-1])
                val_loss += crit(out.reshape(-1, vocab.vocab_size), tgt[:, 1:].reshape(-1)).item()
        val_loss /= len(val_dl)
        
        print(f"[FASE 2] Epoch {epoch+1}: Train {total_loss/len(train_dl):.4f} | Val {val_loss:.4f} | LR {scheduler_ft.get_last_lr()[0]:.2e}")
        
        if val_loss < best_loss_ft:
            best_loss_ft, patience = val_loss, 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  Guardado (val_loss: {best_loss_ft:.4f})")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping en epoch {epoch+1}")
                break
    
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Mejor Val Loss Fase 1: {best_loss:.4f}")
    print(f"Mejor Val Loss Fase 2: {best_loss_ft:.4f}")

def generate(text):
    vocab = Vocabulary()
    vocab.load(Config.VOCAB_PATH)
    tokenizer = AutoTokenizer.from_pretrained(Config.BETO_MODEL)
    
    model = TransformerModel(vocab.vocab_size).to(Config.DEVICE)
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()
    
    # Parámetros de generación
    temperature = 0.7
    top_k = 40
    repetition_penalty = 1.2
    
    tokens = tokenizer(text, return_tensors='pt').to(Config.DEVICE)
    src_ids, src_mask = tokens['input_ids'], tokens['attention_mask']
    
    tgt_idx = [vocab.word2idx['<SOS>']]
    
    for _ in range(Config.MAX_OUTPUT_LEN):
        tgt = torch.tensor([tgt_idx]).to(Config.DEVICE)
        with torch.no_grad():
            out = model(src_ids, src_mask, tgt)
        
        # Obtener logits y aplicar temperatura
        logits = out[0, -1] / temperature
        
        # Penalización de repetición
        for token_id in set(tgt_idx):
            if token_id in [0, 1, 2, 3]: continue
            logits[token_id] /= repetition_penalty

        # Evitar generar <UNK> y <PAD>
        logits[vocab.word2idx['<UNK>']] = -1e9
        logits[vocab.word2idx['<PAD>']] = -1e9
        
        # Filtrado Top-K
        v, idx = torch.topk(logits, top_k)
        probs = torch.softmax(v, dim=-1)
        next_token = idx[torch.multinomial(probs, 1)].item()
        
        if next_token == vocab.word2idx['<EOS>']: break
        tgt_idx.append(next_token)
        
    print("Generado:", vocab.decode(tgt_idx))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--train': train()
    elif len(sys.argv) > 1 and sys.argv[1] == '--demo':
        while True:
            t = input("Tú: ")
            if t=='salir': break
            generate(t)
    else: print("Use --train or --demo")