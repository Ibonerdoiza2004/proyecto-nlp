import ast
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm

# Configuracion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(10)
torch.manual_seed(10)

# Hiperparametros
EMBEDDING_DIM = 768  
NUM_FILTERS = 64
KERNEL_SIZES = [2, 3, 4]
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 16  
EPOCHS = 40
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
MAX_LENGTH = 128

print("CNN + LSTM + BERT (MEAN POOLING)")

# Cargar datos
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

texts = df["text"].tolist()
labels = df["speaker"].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=10, stratify=labels_encoded
)

# Mean pooling
bert_mean_path = os.path.join("models", "bert_mean.npz")
embeddings_npz = np.load(bert_mean_path)
all_embeddings = embeddings_npz[embeddings_npz.files[0]]

# Crear un mapeo directo de texto a embedding para evitar problemas de alineación
text_to_embedding = {}
for i, text in enumerate(df["text"]):
    text_to_embedding[text] = all_embeddings[i]

# Obtener embeddings usando el mapeo directo
X_train_embeddings = np.array([text_to_embedding[text] for text in X_train])
X_test_embeddings = np.array([text_to_embedding[text] for text in X_test])
X_train_embeddings = torch.tensor(X_train_embeddings, dtype=torch.float32)
X_test_embeddings = torch.tensor(X_test_embeddings, dtype=torch.float32)

# Congelar embeddings pre-entrenados
X_train_embeddings.requires_grad_(False)
X_test_embeddings.requires_grad_(False)

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

# Modelo CNN-LSTM con embeddings de BERT (adaptado para mean pooling)
class CNNLSTMBERTClassifier(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_sizes, 
                 lstm_hidden, lstm_layers, num_classes, dropout):
        super(CNNLSTMBERTClassifier, self).__init__()
        
        # Como usamos mean pooling (embeddings 2D), necesitamos expandir la dimensión
        # para simular una "secuencia" de longitud 1
        self.expand_dim = nn.Linear(embedding_dim, embedding_dim * 4)  # Crear una secuencia de 4 tokens
        
        # Multiples CNNs con diferentes kernel sizes
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
        # x shape: [batch_size, embedding_dim]
        
        # Expandir dimensión para crear una "secuencia" artificial
        x = self.expand_dim(x)  # [batch_size, embedding_dim * 4]
        x = x.view(x.size(0), 4, -1)  # [batch_size, 4, embedding_dim]
        x = x.transpose(1, 2)  # [batch_size, embedding_dim, 4] para Conv1d
        
        # Aplicar CNNs
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(x) 
            conv_out = bn(conv_out)
            conv_out = self.relu(conv_out)
            conv_outputs.append(conv_out)
        
        # Encontrar longitud máxima
        max_len = max(out.size(2) for out in conv_outputs)
        
        # Hacer padding a todas las salidas para que tengan la misma longitud máxima
        padded_outputs = []
        for out in conv_outputs:
            if out.size(2) < max_len:
                padding_size = max_len - out.size(2)
                padded = torch.nn.functional.pad(out, (0, padding_size), mode='constant', value=0)
                padded_outputs.append(padded)
            else:
                padded_outputs.append(out)
        
        # Concatenar salidas de CNNs
        cnn_features = torch.cat(padded_outputs, dim=1)
        cnn_features = cnn_features.transpose(1, 2)
        cnn_features = self.dropout(cnn_features)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        
        # Clasificacion
        hidden_concat = self.dropout(hidden_concat)
        output = self.fc(hidden_concat)
        
        return output

# Crear modelo
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
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

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
        torch.save(model.state_dict(), 'models/best_cnnlstm_bert.pth')
        print(f"Mejor modelo guardado (val_acc: {val_acc:.4f})")

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_cnnlstm_bert.pth'))

# Evaluacion
print("EVALUACION")

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
print(f"Mejor Accuracy de Validación: {best_val_acc:.4f}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Matriz de confusion
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

# Graficos
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
