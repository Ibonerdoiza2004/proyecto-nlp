"""
Clasificación de Hablantes usando GRU con BERT embeddings
Arquitectura: BETO embeddings (frozen) → GRU Bidireccional → Dense
Técnicas: BERT embeddings, Bidirectional GRU, Multiple Layers, Gradient Clipping, L2 Regularization
Fuentes: PDF págs 38-40 (GRU bidireccional), BERT como extractor de features
"""

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

print("="*60)
print("GRU BIDIRECCIONAL + BERT (BETO)")
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
df["text"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))

print(f"Total de muestras: {len(df)}")
print(f"Distribución de hablantes:\n{df['speaker'].value_counts()}")

texts = df["text"].tolist()
labels = df["speaker"].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"\nClases: {label_encoder.classes_}")
print(f"Número de clases: {num_classes}")

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

print(f"Train: {len(X_train)} muestras")
print(f"Test: {len(X_test)} muestras")


# Cargar embeddings ya calculados de BETO (mean pooling)
import os
bert_mean_path = os.path.join("models", "bert_mean.npz")
embeddings_npz = np.load(bert_mean_path)
all_embeddings = embeddings_npz[embeddings_npz.files[0]]

# Alinear embeddings con los textos
X_train_idx = df.index[df["text"].isin(X_train)].tolist()
X_test_idx = df.index[df["text"].isin(X_test)].tolist()
X_train_embeddings = torch.tensor(all_embeddings[X_train_idx], dtype=torch.float32)
X_test_embeddings = torch.tensor(all_embeddings[X_test_idx], dtype=torch.float32)
bert_embedding_dim = X_train_embeddings.shape[1]

print(f"BERT embedding dim: {bert_embedding_dim} (loaded from models/bert_mean.npz)")

# Dataset

# Dataset para embeddings
class EmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

train_dataset = EmbeddingsDataset(X_train_embeddings, y_train)
test_dataset = EmbeddingsDataset(X_test_embeddings, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# Modelo GRU para embeddings
class GRUEmbeddingsClassifier(nn.Module):
    def __init__(self, bert_embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super(GRUEmbeddingsClassifier, self).__init__()
        # GRU Bidireccional (PDF pág 38-40)
        self.gru = nn.GRU(
            bert_embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, embeddings):
        # embeddings: (batch, seq_len, bert_dim)
        gru_output, hidden = self.gru(embeddings)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        out = self.dropout(final_hidden)
        logits = self.fc(out)
        return logits

# Instanciar modelo
model = GRUEmbeddingsClassifier(
    bert_embedding_dim=bert_embedding_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

print("\n" + "="*60)
print("ARQUITECTURA DEL MODELO")
print("="*60)
print(f"BERT: {BERT_MODEL} (frozen)")
print(f"GRU: {NUM_LAYERS} capas bidireccionales")
print(f"Hidden dim: {HIDDEN_DIM}")
print(f"\nParámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\n" + "="*60)
print("ENTRENAMIENTO")
print("="*60)

train_losses = []
train_accs = []
test_accs = []

N_REPEAT = 5
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        embeddings_batch, labels = batch
        embeddings_batch = embeddings_batch.to(device)
        labels = torch.tensor(labels).to(device)

        # Expand per-sentence embeddings to pseudo-sequence
        seq_embeddings = embeddings_batch.unsqueeze(1).repeat(1, N_REPEAT, 1)

        optimizer.zero_grad()
        outputs = model(seq_embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        
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
        for batch in test_loader:
            embeddings_batch, labels = batch
            embeddings_batch = embeddings_batch.to(device)
            labels = torch.tensor(labels).to(device)
            seq_embeddings = embeddings_batch.unsqueeze(1).repeat(1, N_REPEAT, 1)
            outputs = model(seq_embeddings)
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
    for batch in test_loader:
        embeddings_batch, labels = batch
        embeddings_batch = embeddings_batch.to(device)
        labels = torch.tensor(labels).to(device)
        seq_embeddings = embeddings_batch.unsqueeze(1).repeat(1, N_REPEAT, 1)
        outputs = model(seq_embeddings)
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
plt.title('Matriz de Confusión - GRU + BERT')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_gru_bert.png', dpi=300)
print("\nMatriz de confusión guardada en: confusion_matrix_gru_bert.png")

# Gráficos
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
plt.savefig('training_gru_bert.png', dpi=300)
print("Gráficos guardados en: training_gru_bert.png")

# Guardar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'label_encoder': label_encoder,
    'hyperparameters': {
        'embedding_source': 'models/bert_mean.npz',
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }
}, 'models/gru_bert.pth')

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"Arquitectura: BiGRU ({NUM_LAYERS} capas) + BERT (frozen)")
print("BERT embeddings source: models/bert_mean.npz")
print(f"Hidden dim: {HIDDEN_DIM}")
print(f"Accuracy final: {accuracy:.4f}")
print(f"Técnicas aplicadas:")
print(f"  - BERT embeddings contextuales (frozen)")
print(f"  - Bidirectional GRU (PDF pág 38-40)")
print(f"  - {NUM_LAYERS} capas apiladas")
print(f"  - Gradient clipping ({GRAD_CLIP})")
print(f"  - L2 regularization (weight_decay={WEIGHT_DECAY})")
