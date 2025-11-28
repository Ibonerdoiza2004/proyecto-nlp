import ast

import numpy as np
import pandas as pd

from gensim.models import Word2Vec

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(10)
torch.manual_seed(10)

# Hiperparámetros
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

print("GRU + Word2Vec")

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

# Filtrar frases muy cortas (menos de 3 palabras)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

# Cargar modelo Word2Vec pre-entrenado
w2v_model = Word2Vec.load("models/w2v.model")
word2vec = w2v_model.wv

# Crear vocabulario: mapeo de palabras a índices
vocab = {word: idx + 1 for idx, word in enumerate(word2vec.index_to_key)}
vocab_size = len(vocab) + 1  # +1 para padding (índice 0)

# Convertir lemmas a secuencias de índices
def lemmas_to_indices(lemmas):
    return [vocab[word] for word in lemmas if word in vocab]

df["sequence"] = df["lemmas_no_stop"].apply(lemmas_to_indices)

# Filtrar secuencias vacías
df = df[df["sequence"].apply(len) > 0].copy()

# Preparar datos
X = df["sequence"].tolist()
y = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)

max_length = max(len(text) for text in X)
embedding_dim = word2vec.vector_size

# Dataset personalizado de PyTorch
class SpeakerDataset(Dataset):
    def __init__(self, sequences, labels, max_length):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        if len(seq) < self.max_length:
            seq = seq + [0] * (self.max_length - len(seq))
        else:
            seq = seq[:self.max_length]
        return torch.LongTensor(seq), torch.LongTensor([self.labels[idx]])

# Crear datasets
train_dataset = SpeakerDataset(X_train, y_train, max_length)
test_dataset = SpeakerDataset(X_test, y_test, max_length)

# Crear dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Crear matriz de embeddings
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]

# Modelo GRU Bidireccional
class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(BiGRUClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embeddings pre-entrenados
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        
        # GRU Bidireccional con múltiples capas
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Clasificador
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # GRU
        gru_out, hidden = self.gru(embedded)
        
        # Concatenar último estado forward y backward
        forward_hidden = hidden[0, :, :]
        backward_hidden = hidden[1, :, :]
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

print(model)
print(f"\nParámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("ENTRENAMIENTO")

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
        labels = labels.to(device).squeeze()
        
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
            labels = labels.to(device).squeeze()
            
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_acc = correct / total
    test_accs.append(test_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}")

# Evaluación final
print("EVALUACIÓN FINAL")

model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.squeeze().cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"\nAccuracy: {accuracy:.4f}")
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

# Guardar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'embedding_matrix': embedding_matrix,
    'vocab': vocab,
    'label_encoder': label_encoder,
    'hyperparameters': {
        'hidden_dim': HIDDEN_DIM,
        'num_layers': NUM_LAYERS,
        'dropout': DROPOUT
    }
}, 'models/bigru_w2v.pth')
