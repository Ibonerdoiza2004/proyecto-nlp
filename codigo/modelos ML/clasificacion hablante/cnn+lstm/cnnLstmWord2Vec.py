"""
Clasificación de Hablantes usando CNN-LSTM Híbrido con Word2Vec
Arquitectura: Word2Vec embeddings → CNN (extracción features locales) → LSTM (secuencial) → Dense
Técnicas: Hybrid CNN-LSTM, Multiple kernels, Bidirectional LSTM, Packed Sequences, Gradient Clipping, L2 Reg
Fuentes: PDF págs 25-30 (CNNs), págs 38-40 (LSTM bidireccional), arquitecturas híbridas
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
NUM_FILTERS = 64  # Filtros por kernel
KERNEL_SIZES = [2, 3, 4]  # Múltiples tamaños de kernel
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # L2 regularization
GRAD_CLIP = 5.0

print("="*60)
print("CNN-LSTM HÍBRIDO + WORD2VEC")
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

# Filtrar frases muy cortas (menos de 3 palabras)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

print(f"Total de muestras: {len(df)}")
print(f"Distribución de hablantes:\n{df['speaker'].value_counts()}")

# Cargar modelo Word2Vec pre-entrenado
print("\nCargando modelo Word2Vec...")
w2v_model = Word2Vec.load("models/w2v.model")
word2vec = w2v_model.wv

# Crear vocabulario: mapeo de palabras a índices
vocab = {word: idx + 1 for idx, word in enumerate(word2vec.index_to_key)}
vocab_size = len(vocab) + 1  # +1 para padding (índice 0)

print(f"Tamaño del vocabulario: {vocab_size}")

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

print(f"\nClases: {label_encoder.classes_}")
print(f"Número de clases: {num_classes}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {len(X_train)} muestras")
print(f"Test: {len(X_test)} muestras")

embedding_dim = word2vec.vector_size
max_length = max(len(text) for text in X)
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

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]

# Modelo CNN-LSTM Híbrido
class CNNLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters, kernel_sizes, lstm_hidden, lstm_layers, num_classes, dropout):
        super(CNNLSTMClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embeddings pre-entrenados (frozen)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        
        # CNN: Múltiples capas convolucionales (PDF pág 25-30)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])
        
        # LSTM Bidireccional (PDF pág 38-40)
        cnn_output_dim = num_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(
            cnn_output_dim,
            lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Clasificador
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)  # *2 por bidireccional
    
    def forward(self, x):
        # Embedding: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # CNN: Transponer para conv1d (batch, embedding_dim, seq_len)
        embedded_t = embedded.permute(0, 2, 1)
        
        # Aplicar cada convolución
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(embedded_t)  # (batch, num_filters, seq_len)
            conv_out = bn(conv_out)
            conv_out = torch.relu(conv_out)
            conv_outputs.append(conv_out)
        
        # Asegurar que todas las salidas de conv tengan la misma longitud de secuencia
        max_seq_len = max(c.shape[2] for c in conv_outputs)
        for i in range(len(conv_outputs)):
            if conv_outputs[i].shape[2] < max_seq_len:
                pad_size = max_seq_len - conv_outputs[i].shape[2]
                conv_outputs[i] = torch.nn.functional.pad(conv_outputs[i], (0, pad_size))
        
        # Concatenar todos los outputs de CNN: (batch, num_filters*len(kernels), seq_len)
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Transponer de vuelta para LSTM: (batch, seq_len, cnn_output_dim)
        cnn_features = concatenated.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        # Concatenar último estado forward y backward
        forward_hidden = hidden[0, :, :]
        backward_hidden = hidden[1, :, :]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Clasificación
        out = self.dropout(final_hidden)
        logits = self.fc(out)
        
        return logits

# Instanciar modelo
model = CNNLSTMClassifier(
    embedding_matrix=embedding_matrix,
    num_filters=NUM_FILTERS,
    kernel_sizes=KERNEL_SIZES,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
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
    weight_decay=WEIGHT_DECAY
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
        all_labels.extend(labels.squeeze().cpu().numpy())

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
plt.title('Matriz de Confusión - CNN-LSTM Híbrido + Word2Vec')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn_lstm_w2v.png', dpi=300)
print("\nMatriz de confusión guardada en: confusion_matrix_cnn_lstm_w2v.png")

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
plt.savefig('training_cnn_lstm_w2v.png', dpi=300)
print("Gráficos guardados en: training_cnn_lstm_w2v.png")

# Guardar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'embedding_matrix': embedding_matrix,
    'vocab': vocab,
    'label_encoder': label_encoder,
    'hyperparameters': {
        'num_filters': NUM_FILTERS,
        'kernel_sizes': KERNEL_SIZES,
        'lstm_hidden': LSTM_HIDDEN,
        'lstm_layers': LSTM_LAYERS,
        'dropout': DROPOUT
    }
}, 'models/cnn_lstm_w2v.pth')

print("\n" + "="*60)
print("MODELO GUARDADO")
print("="*60)
print("Modelo guardado en: models/cnn_lstm_w2v.pth")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print(f"Arquitectura: CNN (kernels {KERNEL_SIZES}) → BiLSTM + Word2Vec")
print(f"Vocabulario: {vocab_size} palabras")
print(f"CNN filters: {NUM_FILTERS} por kernel")
print(f"LSTM hidden: {LSTM_HIDDEN}")
print(f"Accuracy final: {accuracy:.4f}")
print(f"Técnicas aplicadas:")
print(f"  - Arquitectura híbrida CNN-LSTM")
print(f"  - CNN extrae features locales (PDF pág 25-30)")
print(f"  - LSTM captura dependencias secuenciales (PDF pág 38-40)")
print(f"  - Múltiples kernels {KERNEL_SIZES}")
print(f"  - Bidirectional LSTM")
print(f"  - Packed sequences")
print(f"  - Gradient clipping ({GRAD_CLIP})")
print(f"  - L2 regularization (weight_decay={WEIGHT_DECAY})")
