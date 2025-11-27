"""
LSTM para clasificación de hablantes usando embeddings de BERT
Este modelo usa LSTM sobre embeddings pre-calculados de BETO
"""

import ast
import numpy as np
import pandas as pd
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

MAX_SEQ_LENGTH = 50   # Longitud máxima de secuencia (en tokens, no embeddings)
BERT_DIM = 768        # Dimensión de BERT
LSTM_UNITS = 256      # Unidades LSTM
LSTM_LAYERS = 2       # Capas LSTM (págs 38-40 PDF)
BIDIRECTIONAL = True  # LSTM Bidireccional (págs 59-60 PDF)
USE_ATTENTION = True  # Usar atención Bahdanau (págs 64-71 PDF)
DROPOUT = 0.3         # Dropout optimizado
EPOCHS = 100          # Épocas
BATCH_SIZE = 64       # Batch size
LEARNING_RATE = 0.0005  # Learning rate

print("Cargando datos...")
# Cargar dataset preprocesado (para obtener las secuencias de texto)
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

# Cargar embeddings de BERT
print("\nCargando embeddings de BERT...")
bert_data = np.load("models/bert_mean.npz")
bert_embeddings = bert_data['arr_0']  # Shape: (n_samples, 768)

print(f"Shape de embeddings BERT: {bert_embeddings.shape}")

# Crear secuencias: cada palabra se representa con el embedding BERT de toda la frase
# Esto simula tener embeddings contextuales por palabra
def create_bert_sequences(lemmas, bert_emb, max_len):
    """
    Crea una secuencia donde cada token se representa con el embedding BERT de la frase completa.
    En un escenario real, usarías embeddings token-level de BERT.
    """
    seq_len = min(len(lemmas), max_len)
    # Repetir el embedding de la frase para cada token (simplificación)
    sequence = np.tile(bert_emb, (seq_len, 1))  # (seq_len, 768)
    return sequence

# Crear secuencias con embeddings BERT
sequences = []
valid_indices = []
for idx, (lemmas, emb) in enumerate(zip(df["lemmas_no_stop"], bert_embeddings)):
    if len(lemmas) > 0:
        seq = create_bert_sequences(lemmas, emb, MAX_SEQ_LENGTH)
        sequences.append(seq)
        valid_indices.append(idx)

df = df.iloc[valid_indices].reset_index(drop=True)

# Preparar datos
X = sequences  # Lista de arrays de shape (seq_len, 768)
y = df["speaker"].values

print(f"\nTotal de secuencias: {len(X)}")
print(f"Shape de y: {y.shape}")

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

# Dataset personalizado de PyTorch
class BERTSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])

# Función de collate para padding dinámico con longitudes
def collate_fn(batch):
    sequences, labels = zip(*batch)
    
    # Calcular longitudes reales
    lengths = torch.LongTensor([len(seq) for seq in sequences])
    
    # Padding de secuencias
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    labels = torch.cat(labels)
    return sequences_padded, lengths, labels

# Crear datasets
train_dataset = BERTSequenceDataset(X_train, y_train)
test_dataset = BERTSequenceDataset(X_test, y_test)

# Crear dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# Mecanismo de Atención Bahdanau (págs 64-71, pág 7 PDF RNNs_Atencion)
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)
    
    def forward(self, query, keys):
        """
        Args:
            query: [batch, hidden] - último hidden state del LSTM
            keys: [batch, seq_len, hidden] - todos los hidden states
        Returns:
            context: [batch, hidden] - vector de contexto ponderado
            attention_weights: [batch, seq_len, 1] - pesos de atención
        """
        # Expandir query para broadcasting
        query = query.unsqueeze(1)  # [batch, 1, hidden]
        
        # Calcular scores (pág 7 PDF)
        scores = self.Va(torch.tanh(
            self.Wa(query) + self.Ua(keys)
        ))  # [batch, seq_len, 1]
        
        # Attention weights con softmax
        attention_weights = torch.softmax(scores, dim=1)
        
        # Context vector: suma ponderada de los hidden states
        context = torch.bmm(
            attention_weights.permute(0, 2, 1),  # [batch, 1, seq_len]
            keys  # [batch, seq_len, hidden]
        ).squeeze(1)  # [batch, hidden]
        
        return context, attention_weights

# Modelo LSTM mejorado con embeddings de BERT
class BERTLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_size,
                 num_layers=LSTM_LAYERS, bidirectional=BIDIRECTIONAL,
                 use_attention=USE_ATTENTION, dropout_p=0.3):
        """
        LSTM mejorado que procesa secuencias de embeddings BERT
        Con bidireccionalidad (págs 59-60), múltiples capas (págs 38-40)
        y atención Bahdanau (págs 64-71)
        
        Args:
            input_dim (int): dimensión de entrada (768 para BERT)
            hidden_dim (int): tamaño de la dimensión oculta del LSTM
            output_dim (int): número de clases
            batch_size (int): tamaño del batch
            num_layers (int): número de capas LSTM
            bidirectional (bool): usar LSTM bidireccional
            use_attention (bool): usar mecanismo de atención
            dropout_p (float): probabilidad de dropout
        """
        super(BERTLSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM mejorado con múltiples capas y bidireccionalidad
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_p if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Atención Bahdanau si está activada
        lstm_output_size = hidden_dim * self.num_directions
        if use_attention:
            self.attention = BahdanauAttention(lstm_output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Capa fully connected (ajustada para bidireccionalidad)
        self.fc = nn.Linear(lstm_output_size, output_dim)
        
        self.hidden = self.init_hidden()
    
    def init_hidden(self, batch_size=None):
        """Inicializa estados ocultos del LSTM (ajustado para múltiples capas y bidireccionalidad)"""
        if batch_size is None:
            batch_size = self.batch_size
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def forward(self, x_in, lengths=None, apply_softmax=False):
        """
        Forward pass con packed sequences, bidireccionalidad y atención
        
        Args:
            x_in (torch.Tensor): tensor de entrada [batch_size, seq_len, input_dim]
            lengths (torch.Tensor): longitudes reales de secuencias
            apply_softmax (bool): aplicar softmax (False si se usa CrossEntropyLoss)
        Returns:
            prediction_vector: tensor de salida [batch_size, num_classes]
        """
        # Usar packed sequences si se proporcionan longitudes
        if lengths is not None:
            # Ordenar por longitud (descendente)
            lengths_sorted, perm_idx = lengths.sort(0, descending=True)
            x_sorted = x_in[perm_idx]
            
            # Pack sequences
            packed = torch.nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True
            )
            
            # LSTM sobre secuencias packed
            packed_output, (hidden, cell) = self.lstm(packed, self.hidden)
            
            # Unpack
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            
            # Recuperar orden original
            _, unperm_idx = perm_idx.sort(0)
            lstm_out = lstm_out[unperm_idx]
            hidden = hidden[:, unperm_idx, :]
        else:
            # LSTM sin packed sequences
            lstm_out, (hidden, cell) = self.lstm(x_in, self.hidden)
        
        # Para LSTM bidireccional, concatenar estados finales
        if self.bidirectional:
            forward_hidden = hidden[-2, :, :]
            backward_hidden = hidden[-1, :, :]
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
        else:
            last_hidden = hidden[-1, :, :]
        
        # Aplicar atención si está activada (págs 64-71 PDF)
        if self.use_attention:
            context, attention_weights = self.attention(last_hidden, lstm_out)
            # Combinar context vector con último hidden (residual connection)
            combined = context + last_hidden
            combined = self.dropout(combined)
            prediction_vector = self.fc(combined)
        else:
            last_hidden = self.dropout(last_hidden)
            prediction_vector = self.fc(last_hidden)
        
        if apply_softmax:
            prediction_vector = torch.softmax(prediction_vector, dim=1)
        
        return prediction_vector

# Construcción del modelo
print("\nConstruyendo modelo LSTM con embeddings BERT...")
model = BERTLSTMClassifier(
    input_dim=BERT_DIM,
    hidden_dim=LSTM_UNITS,
    output_dim=num_classes,
    batch_size=BATCH_SIZE,
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
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

# Función de entrenamiento (actualizada)
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, lengths, labels in loader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device).squeeze()
        
        # Reiniciar hidden state con tamaño de batch actual
        model.hidden = model.init_hidden(batch_size=sequences.size(0))
        
        optimizer.zero_grad()
        predictions = model(sequences, lengths)
        loss = criterion(predictions, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        pred_classes = torch.argmax(predictions, dim=1)
        correct += (pred_classes == labels).sum().item()
        total += labels.size(0)
    
    return epoch_loss / len(loader), correct / total

# Función de evaluación (actualizada)
def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, lengths, labels in loader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device).squeeze()
            
            # Reiniciar hidden state con tamaño de batch actual
            model.hidden = model.init_hidden(batch_size=sequences.size(0))
            
            predictions = model(sequences, lengths)
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
        torch.save(model.state_dict(), 'models/best_bert_speaker.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping activado en epoch {epoch+1}')
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_bert_speaker.pth'))

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
    for sequences, lengths, labels in test_loader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        # Reiniciar hidden state con tamaño de batch actual
        model.hidden = model.init_hidden(batch_size=sequences.size(0))
        predictions = model(sequences, lengths)
        pred_classes = torch.argmax(predictions, dim=1)
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.squeeze().cpu().numpy())

y_pred_classes = np.array(all_preds)
y_test_array = np.array(all_labels)

# Reporte de clasificación
print("\n" + "="*60)
print("REPORTE DE CLASIFICACIÓN - BERT")
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
plt.title('Matriz de Confusión - BERT Clasificación de Hablantes')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_bert.png', dpi=300, bbox_inches='tight')
print("\nMatriz de confusión guardada en: confusion_matrix_bert.png")

# Gráficas de entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history['train_acc'], label='Train')
axes[0].plot(history['val_acc'], label='Validation')
axes[0].set_title('Accuracy durante el entrenamiento - BERT')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history['train_loss'], label='Train')
axes[1].plot(history['val_loss'], label='Validation')
axes[1].set_title('Loss durante el entrenamiento - BERT')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_bert.png', dpi=300, bbox_inches='tight')
print("Historial de entrenamiento guardado en: training_history_bert.png")

# Guardar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'input_dim': BERT_DIM,
    'hidden_dim': LSTM_UNITS,
    'output_dim': num_classes,
    'dropout': DROPOUT
}, 'models/bert_lstm_speaker_classifier.pth')
print("\nModelo guardado en: models/bert_lstm_speaker_classifier.pth")

# Guardar label encoder
import joblib
joblib.dump(label_encoder, 'models/label_encoder_speaker_bert.joblib')
print("Label encoder guardado en: models/label_encoder_speaker_bert.joblib")

# Función de ejemplo para predecir nuevas frases
def predecir_hablante_bert(sequence, modelo, label_encoder, device):
    """
    Predice el hablante dada una secuencia de embeddings BERT
    
    Args:
        sequence: numpy array de dimensión (seq_len, 768)
        modelo: modelo entrenado
        label_encoder: encoder de etiquetas
        device: dispositivo (cpu/cuda)
    """
    modelo.eval()
    
    # Convertir a tensor
    sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
    
    # Predecir
    with torch.no_grad():
        # Reiniciar hidden state para batch size = 1
        modelo.hidden = modelo.init_hidden(batch_size=1)
        pred = modelo(sequence_tensor)
        pred_proba = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_proba, dim=1).item()
        pred_conf = pred_proba[0][pred_class].item()
    
    hablante = label_encoder.inverse_transform([pred_class])[0]
    
    return hablante, pred_conf

print("\n" + "="*60)
print("EJEMPLOS DE PREDICCIÓN")
print("="*60)

# Tomar algunos ejemplos del test set
n_ejemplos = 5
ejemplos_idx = np.random.choice(len(X_test), min(n_ejemplos, len(X_test)), replace=False)

for idx in ejemplos_idx:
    sequence = X_test[idx]
    texto_real = df.iloc[idx]["text_clean"][:100] if idx < len(df) else "N/A"
    hablante_real = label_encoder.inverse_transform([y_test[idx]])[0]
    
    hablante_pred, confianza = predecir_hablante_bert(
        sequence, model, label_encoder, device
    )
    
    print(f"\nTexto: '{texto_real}...'")
    print(f"Real: {hablante_real} | Predicción: {hablante_pred} (confianza: {confianza:.2%})")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
print(f"\nNOTA: Este modelo usa LSTM sobre embeddings pre-calculados de BERT (mean pooling)")
print("Cada token se representa con el embedding BERT de la frase completa")
