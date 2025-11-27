"""
Clasificación de Hablantes usando GRU con Word2Vec TRAINABLE
Arquitectura: Word2Vec embeddings (entrenables) → GRU Bidireccional (2 capas) → Self-Attention → Dense
Similar a lstmWord2VecTrainable pero con GRU en lugar de LSTM
"""

import ast
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
np.random.seed(42)
torch.manual_seed(42)

# Hiperparámetros
EMBEDDING_DIM = 200
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.4
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
MAX_SEQ_LENGTH = 150

print("="*60)
print("GRU BIDIRECCIONAL + WORD2VEC TRAINABLE")
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

print(f"Total de muestras: {len(df)}")
print(f"Distribución de hablantes:\n{df['speaker'].value_counts()}")

texts = df["lemmas_no_stop"].tolist()
labels = df["speaker"].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Cargar Word2Vec
print("\nCargando Word2Vec...")
w2v_model = Word2Vec.load("models/w2v.model")
word2vec = w2v_model.wv

vocab = {word: idx + 1 for idx, word in enumerate(word2vec.index_to_key)}
vocab_size = len(vocab) + 1

def text_to_indices(text):
    return [vocab.get(word, 0) for word in text]

X_train_seq = [text_to_indices(text) for text in X_train]
X_test_seq = [text_to_indices(text) for text in X_test]

# Crear matriz de embeddings
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, idx in vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]

# Dataset
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), self.labels[idx], len(self.sequences[idx])

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    sequences = [sequences[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]
    
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    if sequences_padded.size(1) > MAX_SEQ_LENGTH:
        sequences_padded = sequences_padded[:, :MAX_SEQ_LENGTH]
        lengths = [min(l, MAX_SEQ_LENGTH) for l in lengths]
    
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)
    return sequences_padded, labels, lengths

train_dataset = TextDataset(X_train_seq, y_train)
test_dataset = TextDataset(X_test_seq, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Modelo GRU Trainable con Attention
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, gru_output):
        attention_scores = self.attention(gru_output).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context = torch.sum(gru_output * attention_weights.unsqueeze(-1), dim=1)
        return context, attention_weights

class GRUTrainableClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, 
                 dropout, pretrained_embeddings=None):
        super(GRUTrainableClassifier, self).__init__()
        
        # Embedding TRAINABLE
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = True  # TRAINABLE
        
        self.embed_layer_norm = nn.LayerNorm(embedding_dim)
        
        # GRU Bidireccional
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.attention = SelfAttention(hidden_dim * 2)
        self.gru_layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, lengths):
        # Embedding
        embedded = self.embedding(x)
        embedded = self.embed_layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        # Packed sequence
        lengths_cpu = lengths.cpu()
        packed_embedded = pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=True)
        
        # GRU
        packed_output, hidden = self.gru(packed_embedded)
        gru_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        gru_out = self.gru_layer_norm(gru_out)
        gru_out = self.dropout(gru_out)
        
        # Attention
        context, attention_weights = self.attention(gru_out)
        context = self.dropout(context)
        
        output = self.fc(context)
        return output, attention_weights

# Crear modelo
print("\n" + "="*60)
print("CONSTRUYENDO MODELO")
print("="*60)

model = GRUTrainableClassifier(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=num_classes,
    dropout=DROPOUT,
    pretrained_embeddings=embedding_matrix
).to(device)

print(model)
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"¿Embeddings entrenables? {model.embedding.weight.requires_grad}")

# Optimizer y loss
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Entrenamiento
def train_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels, lengths in tqdm(loader, desc="Training"):
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(sequences, lengths)
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
        for sequences, labels, lengths in tqdm(loader, desc="Evaluating"):
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            outputs, _ = model(sequences, lengths)
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
        torch.save(model.state_dict(), 'models/best_gru_word2vec_trainable.pth')
        print(f"✓ Mejor modelo guardado")

# Evaluación
model.load_state_dict(torch.load('models/best_gru_word2vec_trainable.pth'))
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels, lengths in test_loader:
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        outputs, _ = model(sequences, lengths)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("\n" + "="*60)
print("RESULTADOS FINALES")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión - GRU + Word2Vec Trainable')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_gru_w2v_trainable.png', dpi=300)
print("\n✓ Matriz guardada")

# Gráficas
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history['train_acc'], label='Train')
axes[0].plot(history['val_acc'], label='Validation')
axes[0].set_title('Accuracy - GRU + Word2Vec Trainable')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_loss'], label='Train')
axes[1].plot(history['val_loss'], label='Validation')
axes[1].set_title('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_gru_w2v_trainable.png', dpi=300)
print("✓ Historial guardado")

print(f"\n✓ Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print("Embeddings de Word2Vec fueron ENTRENADOS (fine-tuned) durante el proceso")
