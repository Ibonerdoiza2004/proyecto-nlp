import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
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
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 50             # Aumentado para dar margen al Early Stopping
LEARNING_RATE = 0.0005  # Reducido para Fine-Tuning
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
PATIENCE = 15            # Para Early Stopping

print("GRU + Word2Vec (FINE-TUNING ACTIVADO)")

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

# Cargar vocabulario común
import pickle
print("Cargando vocabulario común desde models/word2idx.pkl...")
with open("models/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

vocab_size = len(word2idx)

# Convertir lemmas a secuencias de índices
def lemmas_to_indices(lemmas):
    return [word2idx.get(word, 1) for word in lemmas] # 1 is <unk>

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
for word, idx in word2idx.items():
    if word in ['<pad>', '<unk>']: continue
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

# Modelo GRU Bidireccional
class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(BiGRUClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embeddings pre-entrenados
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # --- CAMBIO CRÍTICO: UNFREEZE ---
        self.embedding.weight.requires_grad = True 
        
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

# Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# Entrenamiento
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Funciones de entrenamiento y evaluación
def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in tqdm(loader, desc="Training"):
        sequences = sequences.to(device)
        labels = labels.to(device).squeeze()
        
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
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(loader, desc="Evaluating"):
            sequences = sequences.to(device)
            labels = labels.to(device).squeeze()
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Métricas
    avg_loss = epoch_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1

print("ENTRENAMIENTO CON EARLY STOPPING")

history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
best_val_f1 = 0.0
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, GRAD_CLIP)
    val_loss, val_acc, val_f1 = eval_epoch(model, test_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} (Acc: {val_acc:.4f})")
    
    scheduler.step(val_loss)
    
    # --- EARLY STOPPING ---
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_bigru_w2v.pth')
        print(f"--> Nuevo mejor modelo guardado (F1: {best_val_f1:.4f})")
    else:
        patience_counter += 1
        print(f"--> No mejora. Patience: {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Deteniendo entrenamiento por Early Stopping.")
            break

# Cargar mejor modelo
print("\nCargando mejor modelo para evaluación final...")
model.load_state_dict(torch.load('models/clasificacion_hablantes/best_bigru_w2v.pth'))

# Evaluación final
print("EVALUACIÓN FINAL")

model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device).squeeze()
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(f"Mejor F1 Macro de Validación: {best_val_f1:.4f}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Matriz de Confusión - BiGRU + Word2Vec (F1: {best_val_f1:.2f})')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('imagenes/confusion_matrix_bigru_w2v.png', dpi=300)

# Gráficas
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(history['train_loss'], color=color, label='Train Loss', linestyle='--')
ax1.plot(history['val_loss'], color='orange', label='Val Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('F1 Score (Macro)', color=color)
ax2.plot(history['val_f1'], color=color, label='Val F1')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

plt.title('Training Loss vs Validation F1 Score')
plt.tight_layout()
plt.savefig('imagenes/training_history_bigru_w2v.png', dpi=300)