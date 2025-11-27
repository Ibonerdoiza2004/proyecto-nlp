"""
Clasificación de Hablantes usando CNN-LSTM Híbrido con BERT
Arquitectura: BERT embeddings (frozen) → CNN (extracción features) → LSTM (secuencial) → Dense
Técnicas: Hybrid CNN-LSTM, Multiple kernels, Bidirectional LSTM, BERT frozen embeddings, Gradient Clipping
Fuentes: PDF págs 25-30 (CNNs), págs 38-40 (LSTM), págs 56-60 (BERT)
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
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
np.random.seed(42)
torch.manual_seed(42)

# Hiperparámetros
BERT_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
EMBEDDING_DIM = 768  # Dimensión de BERT
NUM_FILTERS = 64
KERNEL_SIZES = [2, 3, 4]
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 16  # Más pequeño por BERT
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
MAX_LENGTH = 128

print("="*60)
print("CNN-LSTM HÍBRIDO + BERT")
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

# Reconstruir texto original para BERT
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

# Cargar BERT
print("\n" + "="*60)
print("CARGANDO BERT")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModel.from_pretrained(BERT_MODEL).to(device)
bert_model.eval()  # Frozen
for param in bert_model.parameters():
    param.requires_grad = False

print(f"✓ Modelo BERT cargado: {BERT_MODEL}")

# Extraer embeddings de BERT
def get_bert_embeddings_batch(texts, tokenizer, bert_model, device, batch_size=16, max_length=128):
    """Extrae embeddings de BERT en batches"""
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extrayendo embeddings BERT"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenizar
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)
        
        # Obtener embeddings
        with torch.no_grad():
            outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            # Usar todos los tokens (no solo [CLS])
            embeddings = outputs.last_hidden_state  # [batch, seq_len, 768]
        
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)

print("\nExtrayendo embeddings de BERT para train...")
X_train_embeddings = get_bert_embeddings_batch(X_train, tokenizer, bert_model, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)

print("Extrayendo embeddings de BERT para test...")
X_test_embeddings = get_bert_embeddings_batch(X_test, tokenizer, bert_model, device, batch_size=BATCH_SIZE, max_length=MAX_LENGTH)

print(f"\nEmbeddings train: {X_train_embeddings.shape}")
print(f"Embeddings test: {X_test_embeddings.shape}")

# Dataset
class BERTEmbeddingsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def collate_fn(batch):
    embeddings, labels = zip(*batch)
    embeddings = torch.stack(embeddings)
    labels = torch.LongTensor(labels)
    return embeddings, labels

train_dataset = BERTEmbeddingsDataset(X_train_embeddings, y_train)
test_dataset = BERTEmbeddingsDataset(X_test_embeddings, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Modelo CNN-LSTM con embeddings de BERT
class CNNLSTMBERTClassifier(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes, 
                 lstm_hidden, lstm_layers, num_classes, dropout):
        super(CNNLSTMBERTClassifier, self).__init__()
        
        # Múltiples CNNs con diferentes kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                     out_channels=num_filters,
                     kernel_size=k)
            for k in kernel_sizes
        ])
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=num_filters * len(kernel_sizes),
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x: [batch, seq_len, embed_dim] (embeddings de BERT)
        
        # Transponer para Conv1d: [batch, embed_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Aplicar CNNs
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(x)  # [batch, num_filters, seq_len - k + 1]
            conv_out = bn(conv_out)
            conv_out = self.relu(conv_out)
            conv_outputs.append(conv_out)
        
        # Encontrar longitud mínima
        min_len = min(out.size(2) for out in conv_outputs)
        
        # Truncar todas las salidas a la misma longitud
        conv_outputs = [out[:, :, :min_len] for out in conv_outputs]
        
        # Concatenar features: [batch, num_filters * len(kernels), min_len]
        cnn_features = torch.cat(conv_outputs, dim=1)
        
        # Transponer para LSTM: [batch, min_len, num_filters * len(kernels)]
        cnn_features = cnn_features.transpose(1, 2)
        cnn_features = self.dropout(cnn_features)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        # Concatenar últimos hidden states
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Clasificación
        hidden_concat = self.dropout(hidden_concat)
        output = self.fc(hidden_concat)
        
        return output

# Crear modelo
print("\n" + "="*60)
print("CONSTRUYENDO MODELO CNN-LSTM + BERT")
print("="*60)

model = CNNLSTMBERTClassifier(
    embedding_dim=EMBEDDING_DIM,
    num_filters=NUM_FILTERS,
    kernel_sizes=KERNEL_SIZES,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT
).to(device)

print(model)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Optimizer y loss
from sklearn.utils.class_weight import compute_class_weight
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
    
    for embeddings, labels in tqdm(loader, desc="Training"):
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(embeddings)
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
        for embeddings, labels in tqdm(loader, desc="Evaluating"):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            outputs = model(embeddings)
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
        torch.save(model.state_dict(), 'models/best_cnnlstm_bert.pth')
        print(f"✓ Mejor modelo guardado (val_acc: {val_acc:.4f})")

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_cnnlstm_bert.pth'))

# Evaluación final
print("\n" + "="*60)
print("EVALUACIÓN FINAL")
print("="*60)

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for embeddings, labels in test_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        outputs = model(embeddings)
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
plt.title('Matriz de Confusión - CNN-LSTM + BERT')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_cnnlstm_bert.png', dpi=300, bbox_inches='tight')
print("\n✓ Matriz de confusión guardada")

# Gráficas
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history['train_acc'], label='Train', linewidth=2)
axes[0].plot(history['val_acc'], label='Validation', linewidth=2)
axes[0].set_title('Accuracy - CNN-LSTM + BERT')
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
plt.savefig('training_history_cnnlstm_bert.png', dpi=300, bbox_inches='tight')
print("✓ Historial de entrenamiento guardado")

print("\n" + "="*60)
print("✓ ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"Mejor Accuracy de Validación: {best_val_acc:.4f}")
print(f"Test Accuracy Final: {accuracy_score(all_labels, all_preds):.4f}")
