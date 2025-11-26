"""
LSTM para generación de texto imitando el estilo de Miguel Quintana
Este modelo genera texto palabra por palabra basándose en el contexto anterior
"""

import ast
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random

# Configuración
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

# Hiperparámetros
SEQ_LENGTH = 20        # Longitud de secuencia de entrada
EMBEDDING_DIM = 200    # Dimensión de word2vec
LSTM_UNITS = 256       # Unidades LSTM
DROPOUT = 0.2          # Dropout (reducido)
EPOCHS = 100           # Épocas
BATCH_SIZE = 128       # Batch size
LEARNING_RATE = 0.0005 # Learning rate (reducido para estabilidad)

print("Cargando datos...")
# Cargar dataset preprocesado
df = pd.read_csv("dataset/dataset_preprocesado.csv")

# Filtrar solo frases de MIGUEL
df = df[df["speaker"] == "MIGUEL"].copy()
print(f"Total de frases de MIGUEL: {len(df)}")

# Parsear lemmas
def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)

# Filtrar frases muy cortas
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()
print(f"Frases válidas: {len(df)}")

# Cargar modelo Word2Vec pre-entrenado
print("\nCargando modelo Word2Vec...")
w2v_model = Word2Vec.load("models/w2v.model")
word2vec = w2v_model.wv

# Crear vocabulario: mapeo de palabras a índices
vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
idx = 4
for word in word2vec.index_to_key:
    vocab[word] = idx
    idx += 1

vocab_size = len(vocab)
print(f"Tamaño del vocabulario: {vocab_size}")

# Vocabulario inverso (índice -> palabra)
idx_to_word = {idx: word for word, idx in vocab.items()}

# Convertir lemmas a secuencias de índices
def lemmas_to_indices(lemmas):
    return [vocab.get(word, vocab["<UNK>"]) for word in lemmas]

# Crear secuencias de entrenamiento
# Para cada frase, creamos múltiples ejemplos de (contexto -> siguiente palabra)
print("\nCreando secuencias de entrenamiento...")

sequences_X = []  # Secuencias de contexto
sequences_y = []  # Siguiente palabra a predecir

for lemmas in df["lemmas_no_stop"]:
    # Añadir tokens de inicio y fin
    indices = [vocab["<START>"]] + lemmas_to_indices(lemmas) + [vocab["<END>"]]
    
    # Crear secuencias deslizantes
    for i in range(1, len(indices)):
        # Contexto: desde el inicio hasta la posición actual (máximo SEQ_LENGTH)
        start_idx = max(0, i - SEQ_LENGTH)
        context = indices[start_idx:i]
        
        # Padding si es necesario
        if len(context) < SEQ_LENGTH:
            context = [vocab["<PAD>"]] * (SEQ_LENGTH - len(context)) + context
        
        # Target: siguiente palabra
        target = indices[i]
        
        sequences_X.append(context)
        sequences_y.append(target)

sequences_X = np.array(sequences_X)
sequences_y = np.array(sequences_y)

print(f"Total de secuencias de entrenamiento: {len(sequences_X)}")
print(f"Shape de X: {sequences_X.shape}")
print(f"Shape de y: {sequences_y.shape}")

# Split train/validation
split_idx = int(0.9 * len(sequences_X))
X_train, X_val = sequences_X[:split_idx], sequences_X[split_idx:]
y_train, y_val = sequences_y[:split_idx], sequences_y[split_idx:]

print(f"\nTrain: {len(X_train)} secuencias")
print(f"Validation: {len(X_val)} secuencias")

# Crear matriz de embeddings
print("\nCreando matriz de embeddings...")
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

# Tokens especiales con embeddings aleatorios
for token in ["<PAD>", "<UNK>", "<START>", "<END>"]:
    if token in vocab:
        embedding_matrix[vocab[token]] = np.random.randn(EMBEDDING_DIM) * 0.1

# Embeddings de word2vec
for word, idx in vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]

# Dataset personalizado
class TextGenerationDataset(Dataset):
    def __init__(self, sequences_X, sequences_y):
        self.X = torch.LongTensor(sequences_X)
        self.y = torch.LongTensor(sequences_y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Crear datasets
train_dataset = TextGenerationDataset(X_train, y_train)
val_dataset = TextGenerationDataset(X_val, y_val)

# Crear dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False
)

# Modelo LSTM para generación
class LSTMGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, batch_size,
                 dropout_p=0.3, pretrained_embeddings=None, padding_idx=0):
        """
        LSTM para generación de texto
        
        Args:
            vocab_size (int): tamaño del vocabulario
            embedding_dim (int): dimensión de embeddings
            hidden_dim (int): dimensión oculta del LSTM
            batch_size (int): tamaño del batch
            dropout_p (float): probabilidad de dropout
            pretrained_embeddings (numpy.array): embeddings pre-entrenados
            padding_idx (int): índice de padding
        """
        super(LSTMGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        
        # Capa de embedding
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx
            )
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings
            )
            # Los embeddings se pueden entrenar (no congelados)
        
        # LSTM (1 capa para empezar, más estable)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Capa fully connected
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden = self.init_hidden()
    
    def init_hidden(self, batch_size=None):
        """Inicializa estados ocultos del LSTM"""
        if batch_size is None:
            batch_size = self.batch_size
        # 1 capa de LSTM
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def forward(self, x_in):
        """
        Args:
            x_in (torch.Tensor): tensor de entrada [batch_size, seq_len]
        Returns:
            output: tensor de salida [batch_size, vocab_size]
        """
        # Embedding
        embedded = self.embedding(x_in)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded, self.hidden)
        # lstm_out: [batch_size, seq_len, hidden_dim]
        
        # Tomar último timestep
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        last_output = self.dropout(last_output)
        
        # Fully connected
        output = self.fc(last_output)  # [batch_size, vocab_size]
        
        return output

# Construcción del modelo
print("\nConstruyendo modelo LSTM generativo...")
model = LSTMGenerator(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=LSTM_UNITS,
    batch_size=BATCH_SIZE,
    dropout_p=DROPOUT,
    pretrained_embeddings=embedding_matrix
).to(device)

print(model)
print(f"\nParámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Optimizer y loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])

# Learning rate scheduler (más conservador)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
)

# Función de entrenamiento
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        # Reiniciar hidden state
        model.hidden = model.init_hidden(batch_size=sequences.size(0))
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Gradient clipping para estabilidad (menos agresivo)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_classes = torch.argmax(outputs, dim=1)
        correct += (pred_classes == targets).sum().item()
        total += targets.size(0)
    
    return epoch_loss / len(loader), correct / total

# Función de evaluación
def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Reiniciar hidden state
            model.hidden = model.init_hidden(batch_size=sequences.size(0))
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            epoch_loss += loss.item()
            pred_classes = torch.argmax(outputs, dim=1)
            correct += (pred_classes == targets).sum().item()
            total += targets.size(0)
    
    return epoch_loss / len(loader), correct / total

# Entrenamiento
print("\nEntrenando modelo...")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience = 100
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # Learning rate scheduler
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_lstm_generator.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping activado en epoch {epoch+1}')
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_lstm_generator.pth'))

# Guardar modelo completo
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': LSTM_UNITS,
    'dropout': DROPOUT,
    'seq_length': SEQ_LENGTH
}, 'models/lstm_text_generator.pth')
print("\nModelo guardado en: models/lstm_text_generator.pth")

# Guardar vocabulario
with open('models/vocab_generator.pkl', 'wb') as f:
    pickle.dump({
        'vocab': vocab,
        'idx_to_word': idx_to_word,
        'seq_length': SEQ_LENGTH
    }, f)
print("Vocabulario guardado en: models/vocab_generator.pkl")

# Función para generar texto
def generate_text(model, start_text, vocab, idx_to_word, max_length=50, 
                   temperature=1.0, device=device):
    """
    Genera texto a partir de un texto inicial
    
    Args:
        model: modelo entrenado
        start_text: lista de palabras iniciales
        vocab: diccionario palabra -> índice
        idx_to_word: diccionario índice -> palabra
        max_length: longitud máxima a generar
        temperature: controla la aleatoriedad (mayor = más aleatorio)
        device: dispositivo
    
    Returns:
        texto generado como lista de palabras
    """
    model.eval()
    
    # Convertir texto inicial a índices
    context = [vocab.get(word, vocab["<UNK>"]) for word in start_text]
    generated = start_text.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Preparar secuencia de entrada (últimos SEQ_LENGTH tokens)
            if len(context) > SEQ_LENGTH:
                input_seq = context[-SEQ_LENGTH:]
            else:
                input_seq = [vocab["<PAD>"]] * (SEQ_LENGTH - len(context)) + context
            
            # Convertir a tensor
            input_tensor = torch.LongTensor([input_seq]).to(device)
            
            # Reiniciar hidden state
            model.hidden = model.init_hidden(batch_size=1)
            
            # Predecir siguiente palabra
            output = model(input_tensor)
            
            # Aplicar temperature
            output = output / temperature
            probabilities = torch.softmax(output, dim=1)
            
            # Samplear de la distribución
            next_word_idx = torch.multinomial(probabilities, 1).item()
            
            # Detener si se genera <END>
            if next_word_idx == vocab["<END>"]:
                break
            
            # Añadir palabra generada
            next_word = idx_to_word.get(next_word_idx, "<UNK>")
            
            # Evitar tokens especiales en el output
            if next_word not in ["<PAD>", "<UNK>", "<START>"]:
                generated.append(next_word)
                context.append(next_word_idx)
    
    return generated

# Ejemplos de generación
print("\n" + "="*60)
print("EJEMPLOS DE GENERACIÓN DE TEXTO")
print("="*60)

start_texts = [
    ["hoy", "vamos", "analizar"],
    ["real", "madrid"],
    ["entrenador"],
    ["jugador", "temporada"],
    ["pensar"]
]

for start_text in start_texts:
    print(f"\n{'='*60}")
    print(f"Inicio: {' '.join(start_text)}")
    print(f"{'='*60}")
    
    # Generar con diferentes temperaturas
    for temp in [0.5, 0.8, 1.0]:
        generated = generate_text(
            model, start_text, vocab, idx_to_word,
            max_length=30, temperature=temp, device=device
        )
        print(f"\nTemperature {temp}: {' '.join(generated)}")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"\nNOTA: El modelo ha sido entrenado con {len(df)} frases de Miguel Quintana")
print("Usa generate_text() para generar nuevo texto")
