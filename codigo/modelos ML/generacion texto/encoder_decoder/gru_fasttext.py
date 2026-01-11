import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle


# CONFIGURACIÓN
class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_unificado.csv')
    FASTTEXT_PATH = os.path.join(PROJECT_ROOT, 'models', 'fasttext.model')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'generacion_texto')
    MODEL_PATH = os.path.join(MODEL_DIR, 'enc_dec_gru_ft.pt')
    VOCAB_PATH = os.path.join(MODEL_DIR, 'enc_dec_gru_ft_vocab.pkl')
    
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 256
    NUM_LAYERS = 2
    DROPOUT = 0.2
    
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_LEN = 50
    TEACHER_FORCING_RATIO = 0.5
    PATIENCE = 5
    
    # Fase 2: Fine-tuning con embeddings
    FINETUNE_EPOCHS = 20
    FINETUNE_LR = 0.0001
    
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
        if max_len:
            indices = indices[:max_len-1]
        indices = [self.word2idx['<SOS>']] + indices + [self.word2idx['<EOS>']]
        return indices
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<SOS>']]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def save(self, path):
        with open(path, 'wb') as f: pickle.dump(self.__dict__, f)
    
    def load(self, path):
        with open(path, 'rb') as f: self.__dict__.update(pickle.load(f))


# DATASET
class FastTextDataset(Dataset):
    def __init__(self, texts, vocab):
        self.vocab = vocab
        self.pairs = []
        for i in range(len(texts) - 1):
            if len(texts[i].split()) > 5:
                self.pairs.append((texts[i], texts[i + 1]))
                
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        
        src_indices = self.vocab.encode(src, Config.MAX_LEN)
        tgt_indices = self.vocab.encode(tgt, Config.MAX_LEN)
        
        # Padding
        src_indices += [self.vocab.word2idx['<PAD>']] * (Config.MAX_LEN - len(src_indices))
        tgt_indices += [self.vocab.word2idx['<PAD>']] * (Config.MAX_LEN - len(tgt_indices))
        
        return torch.tensor(src_indices[:Config.MAX_LEN]), torch.tensor(tgt_indices[:Config.MAX_LEN])


# MODELO
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        outputs, hidden = self.gru(embedded)
        return hidden
    
    def load_fasttext_embeddings(self, vocab, fasttext_model):
        count = 0
        for word, idx in vocab.word2idx.items():
            if word in fasttext_model.wv:
                self.embedding.weight.data[idx] = torch.tensor(fasttext_model.wv[word])
                count += 1


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(src.device)
        
        hidden = self.encoder(src)
        
        input = tgt[:, 0].unsqueeze(1)
        
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            
            top1 = output.argmax(1)
            
            if np.random.random() < teacher_forcing_ratio:
                input = tgt[:, t].unsqueeze(1)
            else:
                input = top1.unsqueeze(1)
                
        return outputs


# ENTRENAMIENTO
def train():
    print("ENCODER-DECODER: FASTTEXT + GRU")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Cargar datos
    df = pd.read_csv(Config.DATASET_PATH)
    texts = df[df['speaker'] == 'MIGUEL']['text'].tolist()
    
    # Vocabulario
    vocab = Vocabulary()
    vocab.build_vocab(texts)
    vocab.save(Config.VOCAB_PATH)
    
    # Split datos
    split = int(len(texts) * 0.9)
    train_dataset = FastTextDataset(texts[:split], vocab)
    val_dataset = FastTextDataset(texts[split:], vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # Modelo
    model = Seq2Seq(vocab.vocab_size, Config.EMBEDDING_DIM, Config.HIDDEN_DIM, 
                   Config.NUM_LAYERS, Config.DROPOUT).to(Config.DEVICE)
    
    # Cargar FastText
    try:
        from gensim.models import FastText
        if os.path.exists(Config.FASTTEXT_PATH):
            ft_model = FastText.load(Config.FASTTEXT_PATH)
            model.encoder.load_fasttext_embeddings(vocab, ft_model)
            model.decoder.load_fasttext_embeddings(vocab, ft_model)
        else:
            print("No se encontró el modelo FastText, entrenando desde cero.")
    except Exception as e:
        print(f"Error cargando FastText: {e}")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # FASE 1: EMBEDDINGS CONGELADOS
    print("FASE 1: Entrenamiento con embeddings congelados")
    
    # Congelar embeddings
    model.encoder.embedding.weight.requires_grad = False
    model.decoder.embedding.weight.requires_grad = False
    
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Config.LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        # Train
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(Config.DEVICE), tgt.to(Config.DEVICE)
            optimizer.zero_grad()
            output = model(src, tgt)
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(Config.DEVICE), tgt.to(Config.DEVICE)
                output = model(src, tgt, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                val_loss += criterion(output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1)).item()
        val_loss /= len(val_loader)
        
        print(f"[FASE 1] Epoch {epoch + 1}: Train {train_loss:.4f} | Val {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'model': model.state_dict(), 'vocab_size': vocab.vocab_size, 'phase': 1}, Config.MODEL_PATH)
            print(f"  Guardado (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping en época {epoch + 1}")
                break
    
    # FASE 2: FINE-TUNING COMPLETO
    print("FASE 2: Fine-tuning completo (embeddings descongelados)")
    
    # Descongelar embeddings
    model.encoder.embedding.weight.requires_grad = True
    model.decoder.embedding.weight.requires_grad = True
    
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=Config.FINETUNE_LR, weight_decay=0.01)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=Config.FINETUNE_EPOCHS)
    
    best_val_loss_ft = best_val_loss
    patience_counter = 0
    
    for epoch in range(Config.FINETUNE_EPOCHS):
        # Train
        model.train()
        total_loss = 0
        for src, tgt in train_loader:
            src, tgt = src.to(Config.DEVICE), tgt.to(Config.DEVICE)
            optimizer_ft.zero_grad()
            output = model(src, tgt, teacher_forcing_ratio=0.2)
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer_ft.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        scheduler_ft.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(Config.DEVICE), tgt.to(Config.DEVICE)
                output = model(src, tgt, teacher_forcing_ratio=0)
                output_dim = output.shape[-1]
                val_loss += criterion(output[:, 1:].reshape(-1, output_dim), tgt[:, 1:].reshape(-1)).item()
        val_loss /= len(val_loader)
        
        print(f"[FASE 2] Epoch {epoch + 1}: Train {train_loss:.4f} | Val {val_loss:.4f} | LR {scheduler_ft.get_last_lr()[0]:.2e}")
        
        if val_loss < best_val_loss_ft:
            best_val_loss_ft = val_loss
            patience_counter = 0
            torch.save({'model': model.state_dict(), 'vocab_size': vocab.vocab_size, 'phase': 2}, Config.MODEL_PATH)
            print(f"  Guardado (val_loss: {best_val_loss_ft:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping en época {epoch + 1}")
                break
    
    print(f"ENTRENAMIENTO COMPLETADO")
    print(f"Mejor Val Loss Fase 1: {best_val_loss:.4f}")
    print(f"Mejor Val Loss Fase 2: {best_val_loss_ft:.4f}")


def demo():
    vocab = Vocabulary()
    vocab.load(Config.VOCAB_PATH)
    
    model = Seq2Seq(vocab.vocab_size, Config.EMBEDDING_DIM, Config.HIDDEN_DIM, 
                   Config.NUM_LAYERS, Config.DROPOUT).to(Config.DEVICE)
    
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print("DEMO: Generador FastText + GRU")
    print("\nEscribe una frase. 'salir' para terminar.\n")
    
    while True:
        text = input("Tú: ")
        if text.lower() == 'salir': break
        
        src = torch.tensor(vocab.encode(text, Config.MAX_LEN)).unsqueeze(0).to(Config.DEVICE)
        src = F.pad(src, (0, Config.MAX_LEN - src.shape[1]), value=0)
        
        hidden = model.encoder(src)
        input_tok = torch.tensor([[vocab.word2idx['<SOS>']]]).to(Config.DEVICE)
        decoded_indices = []
        
        # Parámetros de generación
        temperature = 0.7
        top_k = 40
        repetition_penalty = 1.2
        
        for _ in range(Config.MAX_LEN):
            output, hidden = model.decoder(input_tok, hidden)
            
            # Aplicar temperatura
            logits = output / temperature
            
            # Penalización de repetición
            for token_id in set(decoded_indices):
                if token_id in [0, 1, 2, 3]: continue
                logits[0, token_id] /= repetition_penalty

            # Evitar generar <UNK> y <PAD>
            logits[0, vocab.word2idx['<UNK>']] = -1e9
            logits[0, vocab.word2idx['<PAD>']] = -1e9
            
            # Sampling Top-K
            v, idx = torch.topk(logits[0], top_k)
            probs = torch.softmax(v, dim=-1)
            token_idx = idx[torch.multinomial(probs, 1)].item()
            
            if token_idx == vocab.word2idx['<EOS>']: break
            decoded_indices.append(token_idx)
            input_tok = torch.tensor([[token_idx]]).to(Config.DEVICE)
            
        print("Generado:", vocab.decode(decoded_indices))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--train': train()
    elif len(sys.argv) > 1 and sys.argv[1] == '--demo': demo()
    else: print("Usa --train o --demo")