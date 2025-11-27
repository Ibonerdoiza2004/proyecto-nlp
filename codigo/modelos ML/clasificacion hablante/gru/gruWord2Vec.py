"""
Clasificación de Hablantes usando GRU Bidireccional con Word2Vec
Arquitectura: Word2Vec embeddings → GRU Bidireccional (2 capas) → Dense
Técnicas: Bidirectional, Multiple Layers, Packed Sequences, Gradient Clipping, L2 Regularization
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
from gensim.models import Word2Vec
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
NUM_LAYERS = 2  # Múltiples capas (PDF pág 38-40)
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2 regularization
GRAD_CLIP = 5.0  # Gradient clipping

print("="*60)
print("GRU BIDIRECCIONAL + WORD2VEC")
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

# Entrenar Word2Vec
print("\n" + "="*60)
print("ENTRENANDO WORD2VEC")
print("="*60)

w2v_model = Word2Vec(
    sentences=X_train,
    vector_size=EMBEDDING_DIM,
    window=5,
    min_count=2,
    workers=4,
    sg=1,  # Skip-gram
    epochs=20
)

vocab_size = len(w2v_model.wv)
print(f"Vocabulario: {vocab_size} palabras")

# Crear matriz de embeddings
embedding_matrix = np.zeros((vocab_size + 2, EMBEDDING_DIM))
word2idx = {"<PAD>": 0, "<UNK>": 1}

for idx, word in enumerate(w2v_model.wv.index_to_key, start=2):
    word2idx[word] = idx
    embedding_matrix[idx] = w2v_model.wv[word]

# UNK como promedio
embedding_matrix[1] = embedding_matrix[2:].mean(axis=0)

print(f"Embedding matrix shape: {embedding_matrix.shape}")

# Dataset
class SpeakerDataset(Dataset):
    def __init__(self, texts, labels, word2idx):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        indices = [self.word2idx.get(word, 1) for word in tokens]  # 1 = <UNK>
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Collate function para packed sequences (PDF pág 78)
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return padded_sequences, lengths, labels

train_dataset = SpeakerDataset(X_train, y_train, word2idx)
test_dataset = SpeakerDataset(X_test, y_test, word2idx)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Modelo GRU Bidireccional
class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(BiGRUClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embeddings pre-entrenados (frozen)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        
        # GRU Bidireccional con múltiples capas (PDF pág 38-40)
        self.gru = nn.GRU(
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
    
    def forward(self, x, lengths):
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        # Packed sequence (PDF pág 78)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # GRU
        packed_output, hidden = self.gru(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
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
model = BiGRUClassifier(
    embedding_matrix=embedding_matrix,
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
    
    for sequences, lengths, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
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
        for sequences, lengths, labels in test_loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences, lengths)
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
    for sequences, lengths, labels in test_loader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        
        outputs = model(sequences, lengths)
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
plt.title('Matriz de Confusión - BiGRU + Word2Vec')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_bigru_w2v.png', dpi=300)
print("\nMatriz de confusión guardada en: confusion_matrix_bigru_w2v.png")

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
plt.savefig('training_bigru_w2v.png', dpi=300)
print("Gráficos guardados en: training_bigru_w2v.png")

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
}, 'models/bigru_w2v.pth')

print("\n" + "="*60)
print("MODELO GUARDADO")
print("="*60)
print("Modelo guardado en: models/bigru_w2v.pth")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"Arquitectura: BiGRU ({NUM_LAYERS} capas) + Word2Vec")
print(f"Vocabulario: {vocab_size} palabras")
print(f"Hidden dim: {HIDDEN_DIM}")
print(f"Accuracy final: {accuracy:.4f}")
print(f"Técnicas aplicadas:")
print(f"  - Bidirectional GRU (PDF pág 38-40)")
print(f"  - {NUM_LAYERS} capas apiladas")
print(f"  - Packed sequences (PDF pág 78)")
print(f"  - Gradient clipping ({GRAD_CLIP})")
print(f"  - L2 regularization (weight_decay={WEIGHT_DECAY})")
