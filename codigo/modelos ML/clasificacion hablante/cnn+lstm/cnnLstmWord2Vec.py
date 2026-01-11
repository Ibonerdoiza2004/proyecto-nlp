import ast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(10)
torch.manual_seed(10)

# Hiperparámetros
NUM_FILTERS = 64
KERNEL_SIZES = [2, 3, 4]
LSTM_HIDDEN = 128
LSTM_LAYERS = 1
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
PATIENCE = 15

print("CNN + LSTM + WORD2VEC")

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

# Cargar modelo Word2Vec pre-entrenado
w2v_model = Word2Vec.load("models/w2v.model")
word2vec = w2v_model.wv

# Cargar vocabulario común
with open("models/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

vocab_size = len(word2idx)

# Convertir lemmas a secuencias de índices
def lemmas_to_indices(lemmas):
    return [word2idx.get(word, 1) for word in lemmas]

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

embedding_dim = word2vec.vector_size
max_length = max(len(text) for text in X)

# Dataset
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
for word, idx in word2idx.items():
    if word in ['<pad>', '<unk>']: continue
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

# Modelo CNN-LSTM
class CNNLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters, kernel_sizes, lstm_hidden, lstm_layers, num_classes, dropout):
        super(CNNLSTMClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        # Embeddings pre-entrenados
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        
        # Descongelar embeddings
        self.embedding.weight.requires_grad = True 
        
        # CNN
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])
        
        # LSTM Bidireccional
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
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # CNN
        embedded_t = embedded.permute(0, 2, 1)
        
        # Aplicar cada convolución
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(embedded_t)
            conv_out = bn(conv_out)
            conv_out = torch.relu(conv_out)
            conv_outputs.append(conv_out)
        
        # Asegurar que todas las salidas de conv tengan la misma longitud de secuencia
        max_seq_len = max(c.shape[2] for c in conv_outputs)
        for i in range(len(conv_outputs)):
            if conv_outputs[i].shape[2] < max_seq_len:
                pad_size = max_seq_len - conv_outputs[i].shape[2]
                conv_outputs[i] = torch.nn.functional.pad(conv_outputs[i], (0, pad_size))
        
        # Concatenar todos los outputs de CNN
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Transponer de vuelta para LSTM
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

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("ENTRENAMIENTO")

train_losses = []
val_f1_scores = []
best_val_f1 = 0.0
patience_counter = 0

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
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluación
    model.eval()
    all_val_preds = []
    all_val_labels = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device).squeeze()
            
            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)
            
            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
    
    # Métricas
    val_acc = accuracy_score(all_val_labels, all_val_preds)
    val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
    val_f1_scores.append(val_f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f} - Val F1: {val_f1:.4f} - Val Acc: {val_acc:.4f}")

    # Early Stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_cnnlstm_w2v.pth')
        print(f"  Nuevo mejor modelo guardado (F1: {best_val_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  No mejora. Patience: {patience_counter}/{PATIENCE}")
    
    if patience_counter >= PATIENCE:
        print("Deteniendo entrenamiento por Early Stopping.")
        break

# Evaluación final
print("\nCARGANDO MEJOR MODELO PARA EVALUACIÓN FINAL")
model.load_state_dict(torch.load('models/clasificacion_hablantes/best_cnnlstm_w2v.pth'))
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        labels = labels.to(device).squeeze()
        
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_predictions)
f1_final = f1_score(all_labels, all_predictions, average='macro')

print(f"\nAccuracy Final: {accuracy:.4f}")
print(f"F1-Score Macro Final: {f1_final:.4f}")

print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_, zero_division=0))

# Matriz de confusión
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Matriz de Confusión - CNN-LSTM + Word2Vec (F1: {f1_final:.2f})')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('imagenes/confusion_matrix_cnnlstm_w2v.png', dpi=300)

# Gráficos de entrenamiento (Loss vs F1)
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
plt.savefig('imagenes/training_history_cnnlstm_w2v.png', dpi=300)