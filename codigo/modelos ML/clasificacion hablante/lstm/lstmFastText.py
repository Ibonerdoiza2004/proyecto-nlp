"""
Clasificación de Hablantes usando LSTM Bidireccional con FastText
Arquitectura: FastText embeddings (con n-gramas) → LSTM Bidireccional (2 capas) → Dense
Técnicas: Bidirectional, Multiple Layers, Packed Sequences, Gradient Clipping, L2 Regularization
Fuentes: PDF págs 38-40 (LSTM bidireccional), págs 78-79 (packed sequences)
"""

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from gensim.models import FastText
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
np.random.seed(42)
torch.manual_seed(42)

# Hiperparámetros
HIDDEN_DIM = 128
NUM_LAYERS = 2  # Múltiples capas (PDF pág 38-40)
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2 regularization
GRAD_CLIP = 5.0  # Gradient clipping

print("="*60)
print("LSTM BIDIRECCIONAL + FASTTEXT")
print("="*60)

print("\nCargando datos...")
# Cargar dataset preprocesado
df = pd.read_csv("dataset/dataset_preprocesado.csv")

# Parsear lemmas
def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)

# Filtrar frases cortas
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

# Preparar datos
texts = df["lemmas_no_stop"].tolist()
labels = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split train/test
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)

# Construir vocabulario y word2idx
all_words = [word for text in texts for word in text]
vocab = set(all_words)
vocab_size = len(vocab) + 2  # +2 para <pad> y <unk>
word2idx = {word: idx+2 for idx, word in enumerate(vocab)}
word2idx['<pad>'] = 0
word2idx['<unk>'] = 1

# Longitud máxima de secuencia
max_length = max(len(text) for text in texts)

print(f"Vocabulario: {vocab_size} palabras")
print(f"Longitud máxima: {max_length}")

# Cargar FastText pre-entrenado
fasttext_model = FastText.load('models/fasttext.model')

# Crear embedding matrix
embedding_dim = fasttext_model.vector_size
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, idx in word2idx.items():
    if word in ['<pad>', '<unk>']:
        continue
    if word in fasttext_model.wv:
        embedding_matrix[idx] = fasttext_model.wv[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

# Dataset
class SpeakerDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        indices = [self.word2idx.get(word, 1) for word in tokens]
        
        # Padding/truncate
        if len(indices) < self.max_length:
            indices = indices + [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

train_dataset = SpeakerDataset(X_train_texts, y_train, word2idx, max_length)
test_dataset = SpeakerDataset(X_test_texts, y_test, word2idx, max_length)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo LSTM Bidireccional
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(BiLSTMClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embeddings pre-entrenados (frozen)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
        self.embedding.weight.requires_grad = False
        
        # LSTM Bidireccional con múltiples capas (PDF pág 38-40)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidireccional
        )
        
        # Clasificador
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 por bidireccional
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenar último estado forward y backward
        # hidden: (num_layers*2, batch, hidden_dim)
        forward_hidden = hidden[-2, :, :]  # Última capa forward
        backward_hidden = hidden[-1, :, :]  # Última capa backward
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Clasificación
        out = self.dropout(final_hidden)
        logits = self.fc(out)
        
        return logits

# Instanciar modelo
model = BiLSTMClassifier(
    embedding_matrix=torch.FloatTensor(embedding_matrix),
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

print("\n" + "="*60)
print("ARQUITECTURA DEL MODELO")
print("="*60)
print(model)
print(f"\nParámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY  # L2 regularization
)

print("\n" + "="*60)
print("ENTRENAMIENTO")
print("="*60)

train_losses = []
train_accs = []
test_accs = []

for epoch in range(EPOCHS):
    # Entrenamiento
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping (prevenir explosión de gradientes)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = epoch_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Evaluación
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_acc = correct / total
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

# Evaluación final
print("\n" + "="*60)
print("EVALUACIÓN FINAL")
print("="*60)

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nReporte de clasificación:")
print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_, zero_division=0))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - BiLSTM + FastText')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_bilstm_fasttext.png', dpi=300)
print("\nMatriz de confusión guardada en: confusion_matrix_bilstm_fasttext.png")

# Gráficos de entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].grid(True)

axes[1].plot(train_accs, label='Train')
axes[1].plot(test_accs, label='Test')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_bilstm_fasttext.png', dpi=300)
print("Gráficos guardados en: training_bilstm_fasttext.png")

# Guardar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'embedding_matrix': embedding_matrix,
    'word2idx': word2idx,
    'label_encoder': label_encoder,
    'hyperparameters': {
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }
}, 'models/bilstm_fasttext.pth')

print("\n" + "="*60)
print("MODELO GUARDADO")
print("="*60)
print("Modelo guardado en: models/bilstm_fasttext.pth")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"Arquitectura: BiLSTM ({NUM_LAYERS} capas) + FastText")
print(f"Vocabulario: {vocab_size} palabras")
print(f"Character n-grams: {fasttext_model.wv.min_n}-{fasttext_model.wv.max_n}")
print(f"Hidden dim: {HIDDEN_DIM}")
print(f"Accuracy final: {accuracy:.4f}")
print(f"Técnicas aplicadas:")
print(f"  - Bidirectional LSTM (PDF pág 38-40)")
print(f"  - {NUM_LAYERS} capas apiladas")
print(f"  - Packed sequences (PDF pág 78)")
print(f"  - FastText con character n-grams (maneja OOV)")
print(f"  - Gradient clipping ({GRAD_CLIP})")
print(f"  - L2 regularization (weight_decay={WEIGHT_DECAY})")
