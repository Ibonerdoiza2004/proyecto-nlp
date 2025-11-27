"""
Clasificación de Hablantes usando CNN con FastText
Arquitectura: FastText embeddings (con n-gramas) → CNN múltiples kernels → Global Max Pooling → Dense
Técnicas: FastText con character n-grams, Multiple kernels, Batch Normalization, Gradient Clipping, L2 Regularization
Fuentes: PDF págs 25-30 (CNNs para texto), FastText con n-gramas de caracteres
"""

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
EMBEDDING_DIM = 100
NUM_FILTERS = 128
KERNEL_SIZES = [2, 3, 4, 5]  # Múltiples tamaños de kernel (PDF pág 25-30)
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2 regularization
GRAD_CLIP = 5.0

print("="*60)
print("CNN + FASTTEXT (CON CHARACTER N-GRAMS)")
print("="*60)

print("\nCargando datos...")
# Cargar dataset
df = pd.read_csv("dataset/dataset_preprocesado.csv")

def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

print(f"Total de muestras: {len(df)}")
print(f"Distribución de hablantes:\n{df['speaker'].value_counts()}")

# Preparar datos
texts = df["lemmas_no_stop"].tolist()
labels = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"\nClases: {label_encoder.classes_}")
print(f"Número de clases: {num_classes}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

print(f"Train: {len(X_train)} muestras")
print(f"Test: {len(X_test)} muestras")

# Entrenar FastText (con character n-grams)
print("\n" + "="*60)
print("ENTRENANDO FASTTEXT (CON CHARACTER N-GRAMS)")
print("="*60)

fasttext_model = FastText(
    sentences=X_train,
    vector_size=EMBEDDING_DIM,
    window=5,
    min_count=2,
    workers=4,
    sg=1,  # Skip-gram
    min_n=3,  # N-gramas mínimos de caracteres
    max_n=6,  # N-gramas máximos de caracteres
    epochs=20
)

vocab_size = len(fasttext_model.wv)
print(f"Vocabulario: {vocab_size} palabras")
print(f"Character n-grams: {fasttext_model.wv.min_n}-{fasttext_model.wv.max_n}")

# Crear matriz de embeddings
embedding_matrix = np.zeros((vocab_size + 2, EMBEDDING_DIM))
word2idx = {"<PAD>": 0, "<UNK>": 1}

for idx, word in enumerate(fasttext_model.wv.index_to_key, start=2):
    word2idx[word] = idx
    embedding_matrix[idx] = fasttext_model.wv[word]

# UNK como promedio
embedding_matrix[1] = embedding_matrix[2:].mean(axis=0)

print(f"Embedding matrix shape: {embedding_matrix.shape}")

# Calcular longitud máxima
max_length = max(len(text) for text in X_train)
print(f"Longitud máxima de secuencia: {max_length}")

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

train_dataset = SpeakerDataset(X_train, y_train, word2idx, max_length)
test_dataset = SpeakerDataset(X_test, y_test, word2idx, max_length)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo CNN
class CNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters, kernel_sizes, num_classes, dropout):
        super(CNNClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embeddings pre-entrenados (frozen)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        
        # Múltiples capas convolucionales con diferentes kernels (PDF pág 25-30)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        # Batch Normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])
        
        # Dropout y clasificador
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, x):
        # Embedding: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # Transponer para conv1d: (batch, embedding_dim, seq_len)
        embedded = embedded.transpose(1, 2)
        
        # Aplicar cada convolución + BatchNorm + ReLU + MaxPool
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            # Conv: (batch, num_filters, seq_len - kernel_size + 1)
            conv_out = conv(embedded)
            # BatchNorm
            conv_out = bn(conv_out)
            # ReLU
            conv_out = torch.relu(conv_out)
            # Global Max Pooling: (batch, num_filters)
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        # Concatenar todos los outputs: (batch, num_filters * len(kernel_sizes))
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout y clasificación
        out = self.dropout(concatenated)
        logits = self.fc(out)
        
        return logits

# Instanciar modelo
model = CNNClassifier(
    embedding_matrix=embedding_matrix,
    num_filters=NUM_FILTERS,
    kernel_sizes=KERNEL_SIZES,
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
        
        # Gradient clipping
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
print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - CNN + FastText')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn_fasttext.png', dpi=300)
print("\nMatriz de confusión guardada en: confusion_matrix_cnn_fasttext.png")

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
plt.savefig('training_cnn_fasttext.png', dpi=300)
print("Gráficos guardados en: training_cnn_fasttext.png")

# Guardar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'embedding_matrix': embedding_matrix,
    'word2idx': word2idx,
    'label_encoder': label_encoder,
    'max_length': max_length,
    'hyperparameters': {
        'num_filters': NUM_FILTERS,
        'kernel_sizes': KERNEL_SIZES,
        'dropout': DROPOUT
    }
}, 'models/cnn_fasttext.pt')

print("\n" + "="*60)
print("MODELO GUARDADO")
print("="*60)
print("Modelo guardado en: models/cnn_fasttext.pt")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"Arquitectura: CNN (kernels {KERNEL_SIZES}) + FastText")
print(f"Vocabulario: {vocab_size} palabras")
print(f"Character n-grams: {fasttext_model.wv.min_n}-{fasttext_model.wv.max_n}")
print(f"Filtros por kernel: {NUM_FILTERS}")
print(f"Accuracy final: {accuracy:.4f}")
print(f"Técnicas aplicadas:")
print(f"  - FastText con character n-grams (3-6)")
print(f"  - Múltiples kernels {KERNEL_SIZES} (PDF pág 25-30)")
print(f"  - Batch Normalization")
print(f"  - Global Max Pooling")
print(f"  - Gradient clipping ({GRAD_CLIP})")
print(f"  - L2 regularization (weight_decay={WEIGHT_DECAY})")
