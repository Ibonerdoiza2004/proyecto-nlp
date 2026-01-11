import os
import math
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import argparse


# CONFIGURACIÓN
class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_unificado.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'generacion_texto')
    MODEL_PATH = os.path.join(MODEL_DIR, 'transformer_generator.pt')
    VOCAB_PATH = os.path.join(MODEL_DIR, 'transformer_vocab.pkl')
    
    # Hiperparámetros del Transformer
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 4
    D_FF = 1024
    DROPOUT = 0.2
    MAX_SEQ_LEN = 128
    
    # Entrenamiento
    BATCH_SIZE = 16
    EPOCHS = 2000
    LEARNING_RATE = 0.0001
    WARMUP_STEPS = 2000
    SEQ_LENGTH = 32
    PATIENCE = 100
    
    # Generación
    TEMPERATURE = 0.75
    TOP_K = 40
    TOP_P = 0.9
    MAX_GEN_LENGTH = 150
    
    # Dispositivo
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# VOCABULARIO
class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token2idx = {}
        self.idx2token = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        # Tokens especiales
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        
        self.token2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        idx = len(special_tokens)
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.token2idx[word] = idx
                idx += 1
        
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)
        
    def encode(self, text, add_special=False):
        words = text.lower().split()
        tokens = [self.token2idx.get(w, self.token2idx['<UNK>']) for w in words]
        if add_special:
            tokens = [self.token2idx['<BOS>']] + tokens + [self.token2idx['<EOS>']]
        return tokens
    
    def decode(self, indices):
        special = {self.token2idx['<PAD>'], self.token2idx['<BOS>'], 
                   self.token2idx['<EOS>'], self.token2idx['<UNK>']}
        words = [self.idx2token.get(idx, '') for idx in indices if idx not in special]
        return ' '.join(words)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))


# DATASET
class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_length):
        self.vocab = vocab
        self.seq_length = seq_length
        self.sequences = []
        
        all_tokens = []
        for text in texts:
            tokens = vocab.encode(text, add_special=True)
            all_tokens.extend(tokens)
        
        # Crear secuencias con solapamiento
        stride = max(1, seq_length // 2)
        for i in range(0, len(all_tokens) - seq_length, stride):
            self.sequences.append(all_tokens[i:i + seq_length + 1])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) < self.seq_length + 1:
            seq = seq + [0] * (self.seq_length + 1 - len(seq))
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


# COMPONENTES DEL TRANSFORMER

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.W_o(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        attn_out = self.self_attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x


# MODELO TRANSFORMER

class TransformerGenerator(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        if mask is None:
            mask = self.generate_causal_mask(seq_len, x.device)
            mask = (mask == 0)
        
        # Embeddings
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pasar por las capas del decoder
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.output_proj(x)
        
        return logits


# FUNCIONES DE ENTRENAMIENTO

def load_miguel_data():
    df = pd.read_csv(Config.DATASET_PATH)
    miguel_df = df[df['speaker'] == 'MIGUEL']
    texts = miguel_df['text'].tolist()
    
    
    return texts


class TransformerScheduler:
    def __init__(self, optimizer, d_model, warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        return self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}, LR: {scheduler._get_lr():.6f}")
    
    return total_loss / len(dataloader)


def train_model():
    print("ENTRENAMIENTO DEL TRANSFORMER")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Cargar datos
    texts = load_miguel_data()
    
    # Crear vocabulario con todas las frases
    vocab = Vocabulary(min_freq=5)
    vocab.build_vocab(texts)
    vocab.save(Config.VOCAB_PATH)
    
    # Split datos
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Datasets
    train_dataset = TextDataset(train_texts, vocab, Config.SEQ_LENGTH)
    val_dataset = TextDataset(val_texts, vocab, Config.SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # Crear modelo
    model = TransformerGenerator(
        vocab_size=vocab.vocab_size,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        d_ff=Config.D_FF,
        max_seq_len=Config.MAX_SEQ_LEN,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerScheduler(optimizer, Config.D_MODEL, Config.WARMUP_STEPS)
    
    # Entrenamiento
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, Config.DEVICE)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab.vocab_size,
                'd_model': Config.D_MODEL,
                'n_heads': Config.N_HEADS,
                'n_layers': Config.N_LAYERS,
                'd_ff': Config.D_FF,
                'max_seq_len': Config.MAX_SEQ_LEN,
                'dropout': Config.DROPOUT
            }, Config.MODEL_PATH)
            print(f"  Modelo guardado (mejor val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping en época {epoch+1}")
                break
        
        # Generar muestra cada 5 épocas
        if (epoch + 1) % 5 == 0:
            sample = generate_text(model, vocab, "yo creo que", max_length=30)
            print(f"  Muestra: {sample}")
    
    print("ENTRENAMIENTO COMPLETADO")


# FUNCIONES DE GENERACIÓN

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, temperature=1.0):
    logits = logits / temperature
    
    # Top-k
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
    
    # Top-p
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float('-inf')
    
    return logits


def generate_text(model, vocab, seed_text, max_length=100, temperature=0.8, top_k=50, top_p=0.9):
    model.eval()
    
    tokens = vocab.encode(seed_text)
    generated_words = seed_text.lower().split()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Preparar entrada
            input_tokens = tokens[-Config.MAX_SEQ_LEN:]
            x = torch.tensor([input_tokens]).to(Config.DEVICE)
            
            # Forward
            logits = model(x)
            next_token_logits = logits[0, -1, :]
            
            # Evitar generar <UNK>
            if '<UNK>' in vocab.token2idx:
                next_token_logits[vocab.token2idx['<UNK>']] = -float('inf')
                
            # Aplicar filtrado
            filtered_logits = top_k_top_p_filtering(
                next_token_logits.clone(), 
                top_k=top_k, 
                top_p=top_p, 
                temperature=temperature
            )
            
            # Muestrear
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # Verificar fin
            if next_token in [vocab.token2idx['<EOS>'], vocab.token2idx['<PAD>']]:
                break
            
            # Añadir token
            tokens.append(next_token)
            word = vocab.idx2token.get(next_token, '')
            if word not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                generated_words.append(word)
    
    return ' '.join(generated_words)


def generate_greedy(model, vocab, seed_text, max_length=100):
    model.eval()
    
    tokens = vocab.encode(seed_text)
    generated_words = seed_text.lower().split()
    
    with torch.no_grad():
        for _ in range(max_length):
            input_tokens = tokens[-Config.MAX_SEQ_LEN:]
            x = torch.tensor([input_tokens]).to(Config.DEVICE)
            
            logits = model(x)
            next_token = logits[0, -1, :].argmax().item()
            
            if next_token in [vocab.token2idx['<EOS>'], vocab.token2idx['<PAD>']]:
                break
            
            tokens.append(next_token)
            word = vocab.idx2token.get(next_token, '')
            if word not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                generated_words.append(word)
    
    return ' '.join(generated_words)


def load_model():
    vocab = Vocabulary()
    vocab.load(Config.VOCAB_PATH)
    
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    
    model = TransformerGenerator(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        n_heads=checkpoint['n_heads'],
        n_layers=checkpoint['n_layers'],
        d_ff=checkpoint['d_ff'],
        max_seq_len=checkpoint['max_seq_len'],
        dropout=checkpoint['dropout']
    ).to(Config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab


# DEMO

def demo():
    print("DEMO: Generador de texto estilo Miguel Quintana (Transformer)")
    
    try:
        model, vocab = load_model()
        print("  Modelo cargado correctamente")
    except FileNotFoundError:
        print("  No se encontró el modelo entrenado.")
        print("  Ejecuta primero: python transformer_generator.py --train")
        return
    
    print("\nEscribe el comienzo de una frase y el modelo la completará.")
    print("  'salir' - Terminar\n")
    
    while True:
        user_input = input("Tu texto: ").strip()
        
        if user_input.lower() == 'salir':
            break
        
        if not user_input:
            print("Por favor, escribe algo.")
            continue
        
        # Parsear comandos
        if user_input.lower().startswith('greedy:'):
            seed = user_input[7:].strip()
            generated = generate_greedy(model, vocab, seed, max_length=Config.MAX_GEN_LENGTH)
            print(f"\nGenerado (greedy): {generated}\n")
        elif user_input.lower().startswith('temp='):
            try:
                temp_end = user_input.index(':')
                temp = float(user_input[5:temp_end])
                seed = user_input[temp_end+1:].strip()
                generated = generate_text(model, vocab, seed, 
                                         max_length=Config.MAX_GEN_LENGTH, 
                                         temperature=temp)
                print(f"\nGenerado (temp={temp}): {generated}\n")
            except:
                print("Formato incorrecto. Usa: temp=0.5:tu texto")
        else:
            generated = generate_text(
                model, vocab, user_input,
                max_length=Config.MAX_GEN_LENGTH,
                temperature=Config.TEMPERATURE,
                top_k=Config.TOP_K,
                top_p=Config.TOP_P
            )
            print(f"\nGenerado: {generated}\n")


# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador Transformer - Estilo Miguel Quintana')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--demo', action='store_true', help='Demo interactiva')
    parser.add_argument('--generate', type=str, help='Generar texto')
    parser.add_argument('--greedy', action='store_true', help='Usar generación greedy')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperatura')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k para muestreo')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) para muestreo')
    parser.add_argument('--max_length', type=int, default=150, help='Longitud máxima')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.demo:
        demo()
    elif args.generate:
        model, vocab = load_model()
        if args.greedy:
            result = generate_greedy(model, vocab, args.generate, max_length=args.max_length)
        else:
            result = generate_text(model, vocab, args.generate,
                                  max_length=args.max_length,
                                  temperature=args.temperature,
                                  top_k=args.top_k,
                                  top_p=args.top_p)
        print(f"Generado: {result}")
    else:
        demo()
