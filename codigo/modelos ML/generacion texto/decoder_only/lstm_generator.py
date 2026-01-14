import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import argparse


# CONFIGURACIÓN
class Config:
    # Rutas
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_unificado.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'generacion_texto')
    MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_generator.pt')
    VOCAB_PATH = os.path.join(MODEL_DIR, 'lstm_vocab.pkl')
    
    # Hiperparámetros del modelo
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    SEQ_LENGTH = 20
    
    TEMPERATURE = 0.7
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
        # Contar palabras
        word_counts = Counter(text.lower().split())
        
        # Tokens especiales
        special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
        
        # Crear mapeos
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        idx = len(special_tokens)
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        
    def encode(self, text, add_special=False):
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in text.lower().split()]
        if add_special:
            indices = [self.word2idx['<START>']] + indices + [self.word2idx['<END>']]
        return indices
    
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
    def __init__(self, texts, vocab, seq_length):
        self.vocab = vocab
        self.seq_length = seq_length
        
        self.encoded = []
        # Si llega un string único (legacy), lo convertimos a lista pero avisamos
        if isinstance(texts, str): 
            texts = [texts]
            
        # Codificar todas las frases añadiendo tokens especiales
        for text in texts:
            self.encoded.extend(vocab.encode(text, add_special=True))
        
        # Crear secuencias
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


# MODELO LSTM
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super(LSTMGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Capas
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
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
        
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        
        logits = self.fc(output)
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h_0, c_0)


# FUNCIONES DE ENTRENAMIENTO
def load_miguel_data():
    df = pd.read_csv(Config.DATASET_PATH)
    
    # Filtrar solo las frases de Miguel
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
    print("ENTRENAMIENTO DEL MODELO LSTM")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    texts = load_miguel_data()
    
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
    train_dataset = TextDataset(train_texts, vocab, Config.SEQ_LENGTH)
    val_dataset = TextDataset(val_texts, vocab, Config.SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Crear modelo
    model = LSTMGenerator(
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
        
        # Entrenar
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        
        # Validar
        val_loss = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR actual: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping
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
                print(f"\n  Early stopping después de {max_patience} epochs sin mejora")
                break
        
        # Generar muestra
        if (epoch + 1) % 5 == 0:
            sample = generate_text(model, vocab, "Yo creo que ", max_length=100)
            print(f"  Muestra: {sample}")
    
    print("ENTRENAMIENTO COMPLETADO")

# FUNCIONES DE GENERACIÓN
def generate_text(model, vocab, seed_text, max_length=100, temperature=0.8):
    model.eval()
    current_seq = vocab.encode(seed_text)
    if '<START>' in vocab.word2idx:
        current_seq = [vocab.word2idx['<START>']] + current_seq
        
    generated = seed_text.split()
    
    
    with torch.no_grad():
        hidden = None
        
        for _ in range(max_length):
            # Preparar entrada
            x = torch.tensor([current_seq[-Config.SEQ_LENGTH:]]).to(Config.DEVICE)
            
            # Forward
            logits, hidden = model(x, hidden)
            logits = logits[0, -1, :] / temperature
            
            # Evitar generar <UNK>
            logits[vocab.word2idx['<UNK>']] = -float('inf')
            
            # Muestrear con top-k
            top_k = 40
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, 1).item()
            next_idx = top_k_indices[sampled_idx].item()
            
            # Decodificar
            next_word = vocab.idx2word.get(next_idx, '')
            
            # Verificar fin
            if next_word in ['<END>', '<PAD>', '']:
                break
            
            generated.append(next_word)
            current_seq.append(next_idx)
    
    return ' '.join(generated)


def generate_text_beam_search(model, vocab, seed_text, max_length=100, beam_width=3):
    model.eval()
    
    initial_seq = vocab.encode(seed_text)
    if '<START>' in vocab.word2idx:
        initial_seq = [vocab.word2idx['<START>']] + initial_seq

    beams = [(initial_seq, 0.0, None)]
    
    
    with torch.no_grad():
        for _ in range(max_length):
            all_candidates = []
            
            for seq, score, hidden in beams:
                x = torch.tensor([seq[-Config.SEQ_LENGTH:]]).to(Config.DEVICE)
                logits, new_hidden = model(x, hidden)
                
                # Evitar <UNK>
                logits_clean = logits[0, -1, :].clone()
                logits_clean[vocab.word2idx['<UNK>']] = -float('inf')

                
                log_probs = torch.log_softmax(logits_clean, dim=-1)
                
                # Top-k candidatos
                top_k_probs, top_k_indices = torch.topk(log_probs, beam_width)
                
                for i in range(beam_width):
                    idx = top_k_indices[i].item()
                    prob = top_k_probs[i].item()
                    
                    new_seq = seq + [idx]
                    new_score = score + prob
                    all_candidates.append((new_seq, new_score, new_hidden))
            
            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_width]
        
        # Decodificar mejor secuencia
        best_seq = beams[0][0]
        return vocab.decode(best_seq)


def load_model():
    # Cargar vocabulario
    vocab = Vocabulary()
    vocab.load(Config.VOCAB_PATH)
    
    # Cargar modelo
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    
    model = LSTMGenerator(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(Config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab


# DEMO INTERACTIVA
def demo():
    print("DEMO (LSTM)")
    
    # Cargar modelo
    try:
        model, vocab = load_model()
        print(" Modelo cargado correctamente")
    except FileNotFoundError:
        print(" No se encontró el modelo entrenado.")
        print("  Ejecuta primero: python lstm_generator.py --train")
        return
    
    print("\nEscribe el comienzo de una frase y el modelo la completará.")
    print("Comandos especiales:")
    print("  'salir' - Terminar")
    print()
    
    while True:
        user_input = input("Tu texto: ").strip()
        
        if user_input.lower() == 'salir':
            break
        
        if not user_input:
            print("Por favor, escribe algo.")
            continue
        
        if user_input.lower().startswith('beam:'):
            seed = user_input[5:].strip()
            generated = generate_text_beam_search(model, vocab, seed, max_length=Config.MAX_GEN_LENGTH)
            print(f"\nGenerado (beam search): {generated}\n")
        else:
            generated = generate_text(
                model, vocab, user_input, 
                max_length=Config.MAX_GEN_LENGTH,
                temperature=Config.TEMPERATURE
            )
            print(f"\nGenerado: {generated}\n")


# MAIN
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generador de texto LSTM - Estilo Miguel Quintana')
    parser.add_argument('--train', action='store_true', help='Entrenar el modelo')
    parser.add_argument('--demo', action='store_true', help='Ejecutar demo interactiva')
    parser.add_argument('--generate', type=str, help='Generar texto a partir de un seed')
    parser.add_argument('--beam', action='store_true', help='Usar beam search')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperatura para generación')
    parser.add_argument('--max_length', type=int, default=100, help='Longitud máxima de generación')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.demo:
        demo()
    elif args.generate:
        model, vocab = load_model()
        if args.beam:
            result = generate_text_beam_search(model, vocab, args.generate, max_length=args.max_length)
        else:
            result = generate_text(model, vocab, args.generate, 
                                  max_length=args.max_length, 
                                  temperature=args.temperature)
        print(f"Generado: {result}")
    else:
        demo()
