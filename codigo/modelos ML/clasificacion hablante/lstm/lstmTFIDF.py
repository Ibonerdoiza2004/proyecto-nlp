"""
Clasificación de Hablantes usando LSTM con TF-IDF
Arquitectura: TF-IDF → Dense (embedding denso) → LSTM Bidireccional → Dense
Técnicas: TF-IDF vectorization, Dense embedding layer, Bidirectional LSTM, Gradient Clipping
Fuentes: PDF págs 15-18 (TF-IDF), págs 38-40 (LSTM bidireccional)
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
np.random.seed(42)
torch.manual_seed(42)

# Hiperparámetros
MAX_FEATURES = 5000  # Máximo número de features TF-IDF
EMBEDDING_DIM = 128  # Dimensión del embedding denso
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0

print("="*60)
print("LSTM BIDIRECCIONAL + TF-IDF")
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

# Convertir a texto para TF-IDF
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

# TF-IDF Vectorization
print("\n" + "="*60)
print("VECTORIZACIÓN TF-IDF")
print("="*60)

tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 2),  # Unigrams y bigrams
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = tfidf_vectorizer.transform(X_test).toarray()

print(f"Vocabulario TF-IDF: {len(tfidf_vectorizer.vocabulary_)} términos")
print(f"Shape train: {X_train_tfidf.shape}")
print(f"Shape test: {X_test_tfidf.shape}")

# Para LSTM necesitamos expandir TF-IDF a secuencias
# Estrategia: repetir el vector TF-IDF para crear una "secuencia" de longitud fija
SEQ_LENGTH = 10  # Longitud artificial de secuencia

def expand_tfidf_to_sequence(tfidf_matrix, seq_length):
    """Expande matriz TF-IDF a formato secuencial repitiendo el vector"""
    n_samples, n_features = tfidf_matrix.shape
    # Repetir el vector TF-IDF seq_length veces
    expanded = np.repeat(tfidf_matrix[:, np.newaxis, :], seq_length, axis=1)
    return expanded

X_train_seq = expand_tfidf_to_sequence(X_train_tfidf, SEQ_LENGTH)
X_test_seq = expand_tfidf_to_sequence(X_test_tfidf, SEQ_LENGTH)

print(f"\nShape train secuencial: {X_train_seq.shape}")  # [n_samples, seq_len, features]
print(f"Shape test secuencial: {X_test_seq.shape}")

# Dataset
class TFIDFDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset = TFIDFDataset(X_train_seq, y_train)
test_dataset = TFIDFDataset(X_test_seq, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo LSTM con TF-IDF
class LSTMTFIDFClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super(LSTMTFIDFClassifier, self).__init__()
        
        # Capa densa para convertir TF-IDF sparse a embedding denso
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM Bidireccional
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Capa de salida
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        
        # Aplicar embedding denso a cada timestep
        batch_size, seq_len, input_dim = x.size()
        x = x.view(-1, input_dim)  # [batch * seq_len, input_dim]
        embedded = self.embedding_layer(x)  # [batch * seq_len, embedding_dim]
        embedded = embedded.view(batch_size, seq_len, -1)  # [batch, seq_len, embedding_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concatenar últimos hidden states
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Dropout y clasificación
        hidden_concat = self.dropout(hidden_concat)
        output = self.fc(hidden_concat)
        
        return output

# Crear modelo
print("\n" + "="*60)
print("CONSTRUYENDO MODELO LSTM + TF-IDF")
print("="*60)

model = LSTMTFIDFClassifier(
    input_dim=X_train_tfidf.shape[1],
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

print(model)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer y loss
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

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
        torch.save(model.state_dict(), 'models/best_lstm_tfidf.pth')
        print(f"✓ Mejor modelo guardado (val_acc: {val_acc:.4f})")

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_lstm_tfidf.pth'))

# Evaluación final
print("\n" + "="*60)
print("EVALUACIÓN FINAL")
print("="*60)

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

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - LSTM + TF-IDF')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_lstm_tfidf.png', dpi=300, bbox_inches='tight')
print("\n✓ Matriz de confusión guardada")

# Gráficas
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history['train_acc'], label='Train', linewidth=2)
axes[0].plot(history['val_acc'], label='Validation', linewidth=2)
axes[0].set_title('Accuracy - LSTM + TF-IDF')
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
plt.savefig('training_history_lstm_tfidf.png', dpi=300, bbox_inches='tight')
print("✓ Historial de entrenamiento guardado")

print("\n" + "="*60)
print("✓ ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"Mejor Accuracy de Validación: {best_val_acc:.4f}")
print(f"Test Accuracy Final: {accuracy_score(all_labels, all_preds):.4f}")
