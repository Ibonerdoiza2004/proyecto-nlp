import ast
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from gensim.models import FastText

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(10)
torch.manual_seed(10)

# Hiperparámetros
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

print("GRU + FastText (CON CHARACTER N-GRAMS)")

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
vocab_size = len(vocab) + 2
word2idx = {word: idx+2 for idx, word in enumerate(vocab)}
word2idx['<pad>'] = 0
word2idx['<unk>'] = 1

# Longitud máxima de secuencia
max_length = max(len(text) for text in texts)

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
        
        if len(indices) < self.max_length:
            indices = indices + [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

train_dataset = SpeakerDataset(X_train_texts, y_train, word2idx, max_length)
test_dataset = SpeakerDataset(X_test_texts, y_test, word2idx, max_length)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo GRU Bidireccional
class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(BiGRUClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape

        # Embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(embedding_matrix)
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
        
        # Fully connected
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        
        # GRU
        output, hidden = self.gru(embedded)
        
        # Concatenar últimos hidden states de ambas direcciones
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Dropout y clasificación
        hidden_concat = self.dropout(hidden_concat)
        output = self.fc(hidden_concat)
        
        return output

# Crear modelo
model = BiGRUClassifier(
    embedding_matrix=torch.FloatTensor(embedding_matrix),
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

print(model)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Optimizer y loss
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Entrenamiento
def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(loader, desc="Training"):
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
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
        for sequences, labels in tqdm(loader, desc="Evaluating"):
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return epoch_loss / len(loader), correct / total

print("ENTRENAMIENTO")

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
        print(f"Mejor modelo guardado (val_acc: {val_acc:.4f})")

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_gru_fasttext.pth'))

# Evaluación final
print("EVALUACIÓN FINAL")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Mejor Accuracy de Validación: {best_val_acc:.4f}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))

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
