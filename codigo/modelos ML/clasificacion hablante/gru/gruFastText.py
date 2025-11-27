"""
Clasificación de Hablantes usando GRU Bidireccional con FastText
Arquitectura: FastText embeddings (char n-grams) → GRU Bidireccional (2 capas) → Dense
Técnicas: Bidirectional GRU, Multiple Layers, Packed Sequences, Gradient Clipping, L2 Regularization
Fuentes: PDF págs 38-40 (GRU bidireccional), págs 78-79 (packed sequences)
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
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2  # Múltiples capas GRU
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2 regularization
GRAD_CLIP = 5.0  # Gradient clipping

print("="*60)
print("GRU BIDIRECCIONAL + FASTTEXT")
print("="*60)

print("\nCargando datos...")
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

# Entrenar FastText con character n-grams
print("\n" + "="*60)
print("ENTRENANDO FASTTEXT (con character n-grams)")
print("="*60)

ft_model = FastText(
    sentences=X_train,
    vector_size=EMBEDDING_DIM,
    window=5,
    min_count=2,
    workers=4,
    sg=1,  # Skip-gram
    min_n=3,  # Character n-grams mínimo
    max_n=6,  # Character n-grams máximo
    epochs=20
)

vocab_size = len(ft_model.wv)
print(f"Vocabulario: {vocab_size} palabras")

# Crear matriz de embeddings
embedding_matrix = np.zeros((vocab_size + 2, EMBEDDING_DIM))
word2idx = {"<PAD>": 0, "<UNK>": 1}

for idx, word in enumerate(ft_model.wv.index_to_key):
    word2idx[word] = idx + 2
    embedding_matrix[idx + 2] = ft_model.wv[word]

# UNK como promedio
embedding_matrix[1] = embedding_matrix[2:].mean(axis=0)

print(f"Matriz de embeddings: {embedding_matrix.shape}")

# Convertir textos a secuencias
def text_to_sequence(text, word2idx, max_len=150):
    sequence = [word2idx.get(word, 1) for word in text]  # 1 = UNK
    return sequence[:max_len]

X_train_seq = [text_to_sequence(text, word2idx) for text in X_train]
X_test_seq = [text_to_sequence(text, word2idx) for text in X_test]

# Dataset
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), self.labels[idx], len(self.sequences[idx])

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    
    # Ordenar por longitud (descendente) para packed_sequence
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    sequences = [sequences[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]
    
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)
    
    return sequences_padded, labels, lengths

train_dataset = TextDataset(X_train_seq, y_train)
test_dataset = TextDataset(X_test_seq, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Modelo GRU Bidireccional
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, 
                 dropout, pretrained_embeddings=None):
        super(BiGRUClassifier, self).__init__()
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = False  # Frozen
        
        # GRU Bidireccional con múltiples capas
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected (hidden_dim * 2 por bidireccional)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, lengths):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # Packed sequence para eficiencia
        lengths_cpu = lengths.cpu()
        packed_embedded = pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=True)
        
        # GRU
        packed_output, hidden = self.gru(packed_embedded)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Concatenar últimos hidden states de ambas direcciones
        # hidden: [num_layers*2, batch, hidden_dim]
        hidden_fwd = hidden[-2, :, :]  # Forward de última capa
        hidden_bwd = hidden[-1, :, :]  # Backward de última capa
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Dropout y clasificación
        hidden_concat = self.dropout(hidden_concat)
        output = self.fc(hidden_concat)
        
        return output

# Crear modelo
print("\n" + "="*60)
print("CONSTRUYENDO MODELO GRU BIDIRECCIONAL")
print("="*60)

model = BiGRUClassifier(
    vocab_size=vocab_size + 2,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT,
    pretrained_embeddings=embedding_matrix
).to(device)

print(model)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Optimizer y loss
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Entrenamiento
def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels, lengths in tqdm(loader, desc="Training"):
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return epoch_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels, lengths in tqdm(loader, desc="Evaluating"):
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return epoch_loss / len(loader), correct / total

print("\n" + "="*60)
print("ENTRENAMIENTO")
print("="*60)

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, GRAD_CLIP)
    val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    
    scheduler.step(val_loss)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_gru_fasttext.pth')
        print(f"✓ Mejor modelo guardado (val_acc: {val_acc:.4f})")

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_gru_fasttext.pth'))

# Evaluación final
print("\n" + "="*60)
print("EVALUACIÓN FINAL")
print("="*60)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels, lengths in test_loader:
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        outputs = model(sequences, lengths)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - GRU Bidireccional + FastText')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_gru_fasttext.png', dpi=300, bbox_inches='tight')
print("\n✓ Matriz de confusión guardada")

# Gráficas de entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history['train_acc'], label='Train', linewidth=2)
axes[0].plot(history['val_acc'], label='Validation', linewidth=2)
axes[0].set_title('Accuracy - GRU Bidireccional + FastText')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_loss'], label='Train', linewidth=2)
axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Loss')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_gru_fasttext.png', dpi=300, bbox_inches='tight')
print("✓ Historial de entrenamiento guardado")

print("\n" + "="*60)
print("✓ ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"Mejor Accuracy de Validación: {best_val_acc:.4f}")
print(f"Test Accuracy Final: {accuracy_score(all_labels, all_preds):.4f}")
