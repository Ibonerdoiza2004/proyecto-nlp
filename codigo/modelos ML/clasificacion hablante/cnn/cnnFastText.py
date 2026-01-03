import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(10)
torch.manual_seed(10)

# Hiperparámetros ACTUALIZADOS
NUM_FILTERS = 128
KERNEL_SIZES = [2, 3, 4, 5]
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005   # Reducido para Fine-Tuning
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
PATIENCE = 15             # Para Early Stopping

print("CNN + FASTTEXT")

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

import pickle

# ... (imports)

# Construir vocabulario y word2idx
print("Cargando vocabulario común desde models/word2idx.pkl...")
with open("models/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

vocab_size = len(word2idx)
# word2idx ya tiene <pad> y <unk>

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

# Modelo CNN
class CNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters, kernel_sizes, num_classes, dropout):
        super(CNNClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embeddings pre-entrenados
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # --- CAMBIO IMPORTANTE: DESCONGELAR (UNFREEZE) ---
        self.embedding.weight.requires_grad = True 
        
        # Capas convolucionales con diferentes kernels
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
        # Embedding
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        # Convoluciones y pooling
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(embedded)
            conv_out = bn(conv_out)
            conv_out = torch.relu(conv_out)
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        # Concatenar todos los outputs
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

print(model)
# Verificar parámetros
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Loss y optimizador
criterion = nn.CrossEntropyLoss()
# El optimizador ahora incluirá los embeddings porque requires_grad=True
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("ENTRENAMIENTO CON EARLY STOPPING")

train_losses = []
val_f1_scores = []

# Variables para Early Stopping
best_val_f1 = 0.0
patience_counter = 0
best_model_path = 'models/clasificacion_hablantes/best_cnn_fasttext.pth'

for epoch in range(EPOCHS):
    # --- ENTRENAMIENTO ---
    model.train()
    epoch_loss = 0
    
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
    
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # --- VALIDACIÓN (CALCULAR F1 MACRO) ---
    model.eval()
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
    
    # Calcular F1 Macro
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_f1_scores.append(val_f1)
    
    print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val F1 (Macro): {val_f1:.4f}")

    # --- CHECKPOINT & EARLY STOPPING ---
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"--> Nuevo mejor modelo guardado (F1: {best_val_f1:.4f})")
    else:
        patience_counter += 1
        print(f"--> No mejora. Patience: {patience_counter}/{PATIENCE}")
        
    if patience_counter >= PATIENCE:
        print("Deteniendo entrenamiento por Early Stopping.")
        break

# --- EVALUACIÓN FINAL DEL MEJOR MODELO ---
print("\nCARGANDO MEJOR MODELO PARA EVALUACIÓN FINAL...")
model.load_state_dict(torch.load(best_model_path))
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

# Métricas finales
accuracy = accuracy_score(all_labels, all_predictions)
f1_macro = f1_score(all_labels, all_predictions, average='macro')

print(f"\nAccuracy Final: {accuracy:.4f}")
print(f"F1-Score Macro Final: {f1_macro:.4f}")
print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_, zero_division=0))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Matriz de Confusión - CNN + FastText (F1: {f1_macro:.2f})')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('imagenes/confusion_matrix_cnn_fasttext.png', dpi=300)

# Gráfico de entrenamiento (Loss vs F1)
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color=color)
ax1.plot(train_losses, color=color, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Validation F1 Score', color=color)
ax2.plot(val_f1_scores, color=color, label='Val F1')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Loss vs Validation F1 Score')
fig.tight_layout()
plt.savefig('imagenes/training_metrics_cnn_fasttext.png', dpi=300)