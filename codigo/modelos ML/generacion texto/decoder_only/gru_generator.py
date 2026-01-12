import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import argparse


# CONFIGURACIÓN
class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_unificado.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'generacion_texto')
    MODEL_PATH = os.path.join(MODEL_DIR, 'gru_generator.pt')
    VOCAB_PATH = os.path.join(MODEL_DIR, 'gru_vocab.pkl')
    
    # Hiperparámetros
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 20
    
    TEMPERATURE = 0.8
    MAX_GEN_LENGTH = 50
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# VOCABULARIO
class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, text):
        word_counts = Counter(text.lower().split())
        
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        idx = len(special_tokens)
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
    def encode(self, text):
        return [self.word2idx.get(w, self.word2idx['<UNK>']) for w in text.lower().split()]
    
    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices 
                       if idx not in [self.word2idx['<PAD>'], self.word2idx['<START>'], self.word2idx['<END>']]])
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'word2idx': self.word2idx, 'idx2word': self.idx2word, 
                        'vocab_size': self.vocab_size, 'min_freq': self.min_freq}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.vocab_size = data['vocab_size']
            self.min_freq = data['min_freq']


# DATASET
class TextDataset(Dataset):
    def __init__(self, text, vocab, seq_length):
        self.vocab = vocab
        self.seq_length = seq_length
        
        self.encoded = vocab.encode(text)
        
        self.sequences = []
        for i in range(0, len(self.encoded) - seq_length):
            seq_in = self.encoded[i:i + seq_length]
            seq_out = self.encoded[i + 1:i + seq_length + 1]
            self.sequences.append((seq_in, seq_out))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_in, seq_out = self.sequences[idx]
        return torch.tensor(seq_in, dtype=torch.long), torch.tensor(seq_out, dtype=torch.long)


# MODELO GRU
class GRUGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(GRUGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        output, hidden = self.gru(embedded, hidden)
        output = self.dropout(output)
        
        logits = self.fc(output)
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)


def load_miguel_data():
    df = pd.read_csv(Config.DATASET_PATH)
    
    # Filtrar las frases de Miguel
    miguel_df = df[df['speaker'] == 'MIGUEL']
    
    # Concatenar todos los textos
    texts = miguel_df['text'].tolist()
    
    return texts


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        
        # Calcular loss
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs, _ = model(inputs)
            
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model():
    print("ENTRENAMIENTO DEL MODELO GRU")
    
    # Crear directorio de modelos
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Cargar datos
    texts = load_miguel_data()
    
    # Split train/val
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    full_train_text = ' '.join(train_texts)
    full_val_text = ' '.join(val_texts)
    
    # Crear vocabulario
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(full_train_text + " " + full_val_text) 
    vocab.save(Config.VOCAB_PATH)
    
    # Crear datasets
    train_dataset = TextDataset(full_train_text, vocab, Config.SEQ_LENGTH)
    val_dataset = TextDataset(full_val_text, vocab, Config.SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Crear modelo
    model = GRUGenerator(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Entrenamiento
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 2
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR actual: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab.vocab_size,
                'embedding_dim': Config.EMBEDDING_DIM,
                'hidden_dim': Config.HIDDEN_DIM,
                'num_layers': Config.NUM_LAYERS,
                'dropout': Config.DROPOUT
            }, Config.MODEL_PATH)
            print(f"  Modelo guardado (mejor val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Generar muestra
        if (epoch + 1) % 5 == 0:
            sample = generate_text(model, vocab, "Yo creo que ", max_length=100)
            print(f"  Muestra: {sample}")
    
    print("ENTRENAMIENTO COMPLETADO")


def generate_text(model, vocab, seed_text, max_length=100, temperature=0.8):
    model.eval()
    
    current_seq = vocab.encode(seed_text)
    generated = seed_text.split()
    
    with torch.no_grad():
        hidden = None
        
        for _ in range(max_length):
            # Preparar entrada
            x = torch.tensor([current_seq[-Config.SEQ_LENGTH:]]).to(Config.DEVICE)
            
            # Forward
            logits, hidden = model(x, hidden)
            logits = logits[0, -1, :] / temperature
            
            # Evitar generar <UNK> y <PAD>
            logits[vocab.word2idx['<UNK>']] = -float('inf')
            logits[vocab.word2idx['<PAD>']] = -float('inf')
            
            # Muestrear
            probs = torch.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            
            # Decodificar
            next_word = vocab.idx2word.get(next_idx, '')
            
            # Verificar fin
            if next_word in ['<END>', '<PAD>', '']:
                break
            
            generated.append(next_word)
            current_seq.append(next_idx)
    
    return ' '.join(generated)


def load_model():
    # Cargar vocabulario
    vocab = Vocabulary()
    vocab.load(Config.VOCAB_PATH)
    
    # Cargar modelo
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    
    model = GRUGenerator(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(Config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab


# DEMO
def demo():
    print("DEMO (GRU)")
    
    # Cargar modelo
    try:
        model, vocab = load_model()
        print("Modelo cargado correctamente")
    except FileNotFoundError:
        print("✗ No se encontró el modelo entrenado.")
        print("  Ejecuta primero: python gru_generator.py --train")
        return
    
    print("\nEscribe el comienzo de una frase y el modelo la completará.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        seed = input("Tu texto: ").strip()
        
        if seed.lower() == 'salir':
            break
        
        if not seed:
            print("Por favor, escribe algo.")
            continue
        
        # Generar texto
        generated = generate_text(
            model, vocab, seed, 
            max_length=Config.MAX_GEN_LENGTH,
            temperature=Config.TEMPERATURE
        )
        
        print(f"\nGenerado: {generated}\n")


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador de texto GRU - Estilo Miguel Quintana')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--demo', action='store_true', help='Ejecutar demo interactiva')
    parser.add_argument('--generate', type=str, help='Generar texto a partir de un seed')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperatura para generación')
    parser.add_argument('--max_length', type=int, default=100, help='Longitud máxima de generación')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.demo:
        demo()
    elif args.generate:
        model, vocab = load_model()
        result = generate_text(model, vocab, args.generate, 
                              max_length=args.max_length, 
                              temperature=args.temperature)
        print(f"Generado: {result}")
    else:
        demo()
