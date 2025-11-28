"""
LSTM para clasificación de hablantes usando Word2Vec
Este modelo clasifica quién dice cada frase en el podcast
"""

import ast
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

MAX_SEQ_LENGTH = 150  # Longitud máxima de secuencia (aumentado)
EMBEDDING_DIM = 200   # Dimensión de word2vec
LSTM_UNITS = 256      # Unidades LSTM (aumentado)
LSTM_LAYERS = 2       # Capas LSTM (mejor representación según pág 38-40 PDF)
DROPOUT = 0.3         # Dropout optimizado para múltiples capas
BIDIRECTIONAL = True  # LSTM Bidireccional (págs 59-60 PDF)
EPOCHS = 100          # Más épocas
BATCH_SIZE = 64       # Batch más grande
LEARNING_RATE = 0.0005  # Learning rate más bajo

print("Cargando datos...")
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
print("\nCreando matriz de embeddings...")
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]

# Modelo LSTM mejorado con técnicas del PDF
class LSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, batch_size, num_layers=LSTM_LAYERS, bidirectional=BIDIRECTIONAL, dropout_p=0.3):
        """
        LSTM Bidireccional con múltiples capas (págs 59-60, 38-40 PDF)
        
        Args:
            vocab_size (int): número de embeddings
            embedding_dim (int): tamaño de los vectores de embedding
            hidden_dim (int): tamaño de la dimensión oculta del LSTM
            output_dim (int): número de clases
            batch_size (int): tamaño del batch
            num_layers (int): número de capas LSTM (págs 38-40 PDF)
            bidirectional (bool): usar LSTM bidireccional (págs 59-60 PDF)
            dropout_p (float): probabilidad de dropout
            pretrained_embeddings (numpy.array): embeddings pre-entrenados (Word2Vec)
        """
        super(LSTMClassifier, self).__init__()
        
        vocab_size, embedding_dim = embedding_matrix.shape
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM mejorado: múltiples capas + bidireccional (págs 38-40, 59-60 PDF)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_p if num_layers > 1 else 0,  # Dropout entre capas
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Capa fully connected (ajustada para bidireccionalidad)
        lstm_output_size = hidden_dim * self.num_directions
        self.fc = nn.Linear(lstm_output_size, output_dim)
    
    def forward(self, x_in):
        """
        Forward pass simplificado sin packed sequences
        
        Args:
            x_in (torch.Tensor): tensor de entrada [batch_size, seq_len]
        Returns:
            prediction_vector: tensor de salida [batch_size, num_classes]
        """
        # Embedding
        embedded = self.embedding(x_in)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout(embedded)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Para LSTM bidireccional, concatenar estados finales de ambas direcciones
        if self.bidirectional:
            # hidden shape: [num_layers * num_directions, batch, hidden_dim]
            # Tomar última capa: forward y backward
            forward_hidden = hidden[-2, :, :]  # Penúltimo: último forward
            backward_hidden = hidden[-1, :, :]  # Último: último backward
            last_output = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            # Para LSTM unidireccional, tomar último hidden state
            last_output = hidden[-1, :, :]  # Última capa
        
        last_output = self.dropout(last_output)
        
        # Fully connected
        prediction_vector = self.fc(last_output)
        
        return prediction_vector

# Construcción del modelo
print("\nConstruyendo modelo LSTM...")
model = LSTMClassifier(
    embedding_matrix=embedding_matrix,
    hidden_dim=LSTM_UNITS,
    output_dim=num_classes,
    batch_size=BATCH_SIZE,
    num_layers=LSTM_LAYERS,
    bidirectional=BIDIRECTIONAL,
    dropout_p=DROPOUT
).to(device)

print(model)
print(f"\nParámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Calcular class weights para balancear el dataset
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"\nClass weights: {dict(zip(label_encoder.classes_, class_weights))}")

# Optimizer y loss (con regularización L2)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Función de entrenamiento (simplificada)
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device).squeeze()
        
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()
        
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_classes = torch.argmax(predictions, dim=1)
        correct += (pred_classes == labels).sum().item()
        total += labels.size(0)
    
    return epoch_loss / len(loader), correct / total

# Función de evaluación (simplificada)
def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device).squeeze()
            
            predictions = model(sequences)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            pred_classes = torch.argmax(predictions, dim=1)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)
    
    return epoch_loss / len(loader), correct / total

# Entrenamiento
print("\nEntrenando modelo...")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience = 60
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # Learning rate scheduler
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_lstm_speaker.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping activado en epoch {epoch+1}')
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_lstm_speaker.pth'))

# Evaluación final
print("\nEvaluando modelo...")
test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Predicciones para matriz de confusión
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        predictions = model(sequences)
        pred_classes = torch.argmax(predictions, dim=1)
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.squeeze().cpu().numpy())

y_pred_classes = np.array(all_preds)
y_test_array = np.array(all_labels)

# Reporte de clasificación
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN")
print("="*60)
print(classification_report(
    y_test_array, y_pred_classes,
    target_names=label_encoder.classes_
))

# Matriz de confusión
cm = confusion_matrix(y_test_array, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Matriz de Confusión - Clasificación de Hablantes')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_lstm.png', dpi=300, bbox_inches='tight')
print("\nMatriz de confusión guardada en: confusion_matrix_lstm.png")

# Gráficas de entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history['train_acc'], label='Train')
axes[0].plot(history['val_acc'], label='Validation')
axes[0].set_title('Accuracy durante el entrenamiento')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history['train_loss'], label='Train')
axes[1].plot(history['val_loss'], label='Validation')
axes[1].set_title('Loss durante el entrenamiento')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_lstm.png', dpi=300, bbox_inches='tight')
print("Historial de entrenamiento guardado en: training_history_lstm.png")

# Guardar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': LSTM_UNITS,
    'output_dim': num_classes,
    'dropout': DROPOUT,
    'num_layers': LSTM_LAYERS,
    'bidirectional': BIDIRECTIONAL
}, 'models/lstm_speaker_classifier.pth')
print("\nModelo guardado en: models/lstm_speaker_classifier.pth")