"""
LSTM MEJORADO para clasificación de hablantes usando Word2Vec (embeddings entrenables)
Este modelo clasifica quién dice cada frase, permitiendo que los embeddings se ajusten

MEJORAS IMPLEMENTADAS:
1. LSTM Bidireccional (2 capas) - captura contexto en ambas direcciones
2. Packed sequences - manejo eficiente de secuencias de distinta longitud
3. Self-Attention - ponderar importancia de cada timestep
4. Layer Normalization - estabiliza entrenamiento
5. Gradient Clipping - evita explosión de gradientes
6. Warmup Learning Rate - estabiliza inicio del entrenamiento
7. Label Smoothing - regularización para evitar overconfidence
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Configuración
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")

MAX_SEQ_LENGTH = 150  # Longitud máxima de secuencia
EMBEDDING_DIM = 200   # Dimensión de word2vec
LSTM_UNITS = 256      # Unidades LSTM por dirección
LSTM_LAYERS = 2       # Número de capas LSTM
DROPOUT = 0.4         # Dropout
EPOCHS = 500          # Épocas
BATCH_SIZE = 64       # Batch size
LEARNING_RATE = 0.001  # Learning rate inicial (con warmup)
WARMUP_STEPS = 300    # Pasos de warmup para learning rate
GRAD_CLIP = 5.0       # Gradient clipping
LABEL_SMOOTHING = 0.1 # Label smoothing para regularización

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

# Dataset personalizado de PyTorch
class SpeakerDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Devolver secuencia, etiqueta y longitud original
        seq = self.sequences[idx]
        return torch.LongTensor(seq), torch.LongTensor([self.labels[idx]]), len(seq)

# Función de collate para padding dinámico con longitudes
def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    
    # Ordenar por longitud (descendente) para packed_sequence
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    
    sequences = [sequences[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]
    
    # Padding de secuencias
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Truncar si es necesario
    if sequences_padded.size(1) > MAX_SEQ_LENGTH:
        sequences_padded = sequences_padded[:, :MAX_SEQ_LENGTH]
        lengths = [min(l, MAX_SEQ_LENGTH) for l in lengths]
    
    labels = torch.cat(labels)
    lengths = torch.LongTensor(lengths)
    
    return sequences_padded, labels, lengths

# Crear datasets
train_dataset = SpeakerDataset(X_train, y_train)
test_dataset = SpeakerDataset(X_test, y_test)

# Crear dataloaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
)

# Crear matriz de embeddings
print("\nCreando matriz de embeddings...")
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, idx in vocab.items():
    if word in word2vec:
        embedding_matrix[idx] = word2vec[word]

# Self-Attention Layer
class SelfAttention(nn.Module):
    """
    Self-Attention para ponderar la importancia de cada timestep en la secuencia
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: [batch_size, seq_len, hidden_dim]
        Returns:
            context: [batch_size, hidden_dim] - representación ponderada
            attention_weights: [batch_size, seq_len] - pesos de atención
        """
        # Calcular scores de atención
        attention_scores = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_scores = attention_scores.squeeze(-1)  # [batch_size, seq_len]
        
        # Aplicar softmax para obtener pesos
        attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, seq_len]
        
        # Ponderar outputs del LSTM
        attention_weights_expanded = attention_weights.unsqueeze(-1)  # [batch_size, seq_len, 1]
        context = torch.sum(lstm_output * attention_weights_expanded, dim=1)  # [batch_size, hidden_dim]
        
        return context, attention_weights


# Modelo LSTM MEJORADO con embeddings entrenables
class LSTMClassifierTrainable(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2,
                 dropout_p=0.3, pretrained_embeddings=None, padding_idx=0):
        """
        LSTM Bidireccional con Self-Attention y Layer Normalization
        
        Args:
            vocab_size (int): número de embeddings
            embedding_dim (int): tamaño de los vectores de embedding
            hidden_dim (int): tamaño de la dimensión oculta del LSTM (por dirección)
            output_dim (int): número de clases
            num_layers (int): número de capas LSTM
            dropout_p (float): probabilidad de dropout
            pretrained_embeddings (numpy.array): embeddings pre-entrenados (Word2Vec)
            padding_idx (int): índice que representa padding
        """
        super(LSTMClassifierTrainable, self).__init__()
        
        # Capa de embedding (SIN CONGELAR - se entrena)
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx
            )
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                _weight=pretrained_embeddings
            )
            # NO congelar embeddings - permitir que se entrenen
            self.embedding.weight.requires_grad = True
        
        # Layer normalization después de embeddings
        self.embed_layer_norm = nn.LayerNorm(embedding_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM BIDIRECCIONAL con múltiples capas
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0,  # Dropout entre capas
            bidirectional=True  # BIDIRECCIONAL
        )
        
        # Self-Attention sobre outputs del LSTM
        # hidden_dim * 2 porque es bidireccional
        self.attention = SelfAttention(hidden_dim * 2)
        
        # Layer normalization después de LSTM
        self.lstm_layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Capas fully connected con residual connection
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_layer_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x_in, lengths=None, apply_softmax=False):
        """
        Args:
            x_in (torch.Tensor): tensor de entrada [batch_size, seq_len]
            lengths (torch.Tensor): longitudes reales de cada secuencia
            apply_softmax (bool): aplicar softmax (False si se usa CrossEntropyLoss)
        Returns:
            prediction_vector: tensor de salida [batch_size, num_classes]
            attention_weights: pesos de atención [batch_size, seq_len]
        """
        batch_size = x_in.size(0)
        
        # Embedding con layer normalization
        embedded = self.embedding(x_in)  # [batch_size, seq_len, embedding_dim]
        embedded = self.embed_layer_norm(embedded)
        embedded = self.dropout(embedded)
        
        # Packed sequence para eficiencia (si se proporcionan longitudes)
        if lengths is not None:
            # Asegurar que lengths esté en CPU para pack_padded_sequence
            lengths_cpu = lengths.cpu()
            packed_embedded = pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=True
            )
            
            # LSTM
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # Unpack
            lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            # Sin packed sequence
            lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # lstm_out: [batch_size, seq_len, hidden_dim * 2] (bidireccional)
        
        # Layer normalization
        lstm_out = self.lstm_layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        
        # Self-Attention para ponderar timesteps
        context, attention_weights = self.attention(lstm_out)
        # context: [batch_size, hidden_dim * 2]
        
        # Dropout después de attention
        context = self.dropout(context)
        
        # Capas fully connected con activación
        fc1_out = self.fc1(context)  # [batch_size, hidden_dim]
        fc1_out = self.fc_layer_norm(fc1_out)
        fc1_out = self.relu(fc1_out)
        fc1_out = self.dropout(fc1_out)
        
        # Salida final
        prediction_vector = self.fc2(fc1_out)  # [batch_size, num_classes]
        
        if apply_softmax:
            prediction_vector = torch.softmax(prediction_vector, dim=1)
        
        return prediction_vector, attention_weights

# Construcción del modelo MEJORADO
print("\n" + "="*60)
print("CONSTRUYENDO MODELO LSTM MEJORADO")
print("="*60)
print("Mejoras implementadas:")
print("  - LSTM Bidireccional (2 capas)")
print("  - Self-Attention")
print("  - Layer Normalization")
print("  - Packed Sequences")
print("  - Gradient Clipping")
print("  - Warmup Learning Rate")
print("  - Label Smoothing")
print("="*60)

model = LSTMClassifierTrainable(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=LSTM_UNITS,
    output_dim=num_classes,
    num_layers=LSTM_LAYERS,
    dropout_p=DROPOUT,
    pretrained_embeddings=embedding_matrix
).to(device)

print(model)
print(f"\nParámetros totales: {sum(p.numel() for p in model.parameters()):,}")
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Verificar que los embeddings son entrenables
print(f"\n¿Embeddings entrenables? {model.embedding.weight.requires_grad}")

# Calcular class weights para balancear el dataset
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
print(f"\nClass weights: {dict(zip(label_encoder.classes_, class_weights))}")

# Optimizer y loss con Label Smoothing
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=LABEL_SMOOTHING)

# Learning rate scheduler (ReduceLROnPlateau)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# Warmup scheduler personalizado
class WarmupScheduler:
    """Learning rate warmup para estabilizar el inicio del entrenamiento"""
    def __init__(self, optimizer, warmup_steps, initial_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.current_step = 0
    
    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup_steps:
            lr = self.initial_lr * (self.current_step / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
    
    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.initial_lr * (self.current_step / self.warmup_steps)
        return self.optimizer.param_groups[0]['lr']

warmup_scheduler = WarmupScheduler(optimizer, WARMUP_STEPS, LEARNING_RATE)

# Función de entrenamiento MEJORADA
def train_epoch(model, loader, optimizer, criterion, device, warmup_scheduler, grad_clip):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for sequences, labels, lengths in loader:
        sequences = sequences.to(device)
        labels = labels.to(device).squeeze()
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass con packed sequences
        predictions, attention_weights = model(sequences, lengths=lengths)
        
        loss = criterion(predictions, labels)
        loss.backward()
        
        # Gradient clipping para evitar explosión de gradientes
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        # Warmup del learning rate
        warmup_scheduler.step()
        
        epoch_loss += loss.item()
        pred_classes = torch.argmax(predictions, dim=1)
        correct += (pred_classes == labels).sum().item()
        total += labels.size(0)
    
    return epoch_loss / len(loader), correct / total

# Función de evaluación MEJORADA
def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels, lengths in loader:
            sequences = sequences.to(device)
            labels = labels.to(device).squeeze()
            lengths = lengths.to(device)
            
            # Forward pass con packed sequences
            predictions, attention_weights = model(sequences, lengths=lengths)
            
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            pred_classes = torch.argmax(predictions, dim=1)
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)
    
    return epoch_loss / len(loader), correct / total

# Entrenamiento con mejoras
print("\n" + "="*60)
print("INICIANDO ENTRENAMIENTO")
print("="*60)
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
best_val_loss = float('inf')
patience = 60
patience_counter = 0

for epoch in range(EPOCHS):
    # Entrenar con gradient clipping y warmup
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, criterion, device, warmup_scheduler, GRAD_CLIP
    )
    val_loss, val_acc = eval_epoch(model, test_loader, criterion, device)
    
    # Guardar historial
    current_lr = warmup_scheduler.get_lr()
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['lr'].append(current_lr)
    
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    print(f'  Learning Rate: {current_lr:.6f}')
    
    # Learning rate scheduler (después del warmup)
    if warmup_scheduler.current_step >= WARMUP_STEPS:
        scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_lstm_speaker_trainable.pth')
        print(f'  ✓ Mejor modelo guardado (val_loss: {val_loss:.4f})')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\n⚠ Early stopping activado en epoch {epoch+1}')
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_lstm_speaker_trainable.pth'))

# Evaluación final
print("\nEvaluando modelo...")
test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Predicciones para matriz de confusión
model.eval()
all_preds = []
all_labels = []
all_attention_weights = []

with torch.no_grad():
    for sequences, labels, lengths in test_loader:
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        
        # Forward pass
        predictions, attention_weights = model(sequences, lengths=lengths)
        pred_classes = torch.argmax(predictions, dim=1)
        
        all_preds.extend(pred_classes.cpu().numpy())
        all_labels.extend(labels.squeeze().cpu().numpy())
        all_attention_weights.append(attention_weights.cpu().numpy())

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
plt.title('Matriz de Confusión - LSTM con Embeddings Entrenables')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_lstm_trainable.png', dpi=300, bbox_inches='tight')
print("\nMatriz de confusión guardada en: confusion_matrix_lstm_trainable.png")

# Gráficas de entrenamiento mejoradas
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Accuracy
axes[0].plot(history['train_acc'], label='Train', linewidth=2)
axes[0].plot(history['val_acc'], label='Validation', linewidth=2)
axes[0].set_title('Accuracy - LSTM Bidireccional con Attention', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history['train_loss'], label='Train', linewidth=2)
axes[1].plot(history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Loss - Label Smoothing + Gradient Clipping', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning Rate (con warmup)
axes[2].plot(history['lr'], label='Learning Rate', linewidth=2, color='green')
axes[2].set_title('Learning Rate Schedule (con Warmup)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Época')
axes[2].set_ylabel('Learning Rate')
axes[2].legend()
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=WARMUP_STEPS // (len(train_loader)), color='red', linestyle='--', 
                label=f'Fin Warmup', alpha=0.5)

plt.tight_layout()
plt.savefig('training_history_lstm_trainable.png', dpi=300, bbox_inches='tight')
print("Historial de entrenamiento guardado en: training_history_lstm_trainable.png")

# Guardar modelo con configuración completa
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': LSTM_UNITS,
    'num_layers': LSTM_LAYERS,
    'output_dim': num_classes,
    'dropout': DROPOUT,
    'bidirectional': True,
    'with_attention': True
}, 'models/lstm_speaker_classifier_trainable.pth')
print("\nModelo guardado en: models/lstm_speaker_classifier_trainable.pth")

# Guardar label encoder
import joblib
joblib.dump(label_encoder, 'models/label_encoder_speaker_trainable.joblib')
print("Label encoder guardado en: models/label_encoder_speaker_trainable.joblib")

# Guardar vocabulario
import pickle
with open('models/vocab_lstm_trainable.pkl', 'wb') as f:
    pickle.dump({'vocab': vocab, 'max_seq_length': MAX_SEQ_LENGTH}, f)
print("Vocabulario guardado en: models/vocab_lstm_trainable.pkl")

# Función de ejemplo para predecir nuevas frases CON ATTENTION
def predecir_hablante(frase, modelo, word2vec, vocab, label_encoder, device, max_seq_length=MAX_SEQ_LENGTH):
    """
    Predice el hablante de una nueva frase y devuelve los pesos de atención
    """
    try:
        import spacy
        nlp = spacy.load('es_core_news_sm')
        
        modelo.eval()
        
        # Preprocesar frase
        doc = nlp(frase.lower())
        lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    except:
        # Si no está spacy, usar preprocesado simple
        modelo.eval()
        lemmas = frase.lower().split()
    
    # Convertir a secuencia de índices
    sequence = [vocab[word] for word in lemmas if word in vocab]
    
    if len(sequence) == 0:
        return None, None, None, None
    
    # Truncar si es necesario
    if len(sequence) > max_seq_length:
        sequence = sequence[:max_seq_length]
    
    # Convertir a tensor
    sequence_tensor = torch.LongTensor([sequence]).to(device)
    length_tensor = torch.LongTensor([len(sequence)]).to(device)
    
    # Predecir
    with torch.no_grad():
        pred, attention_weights = modelo(sequence_tensor, lengths=length_tensor)
        pred_proba = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_proba, dim=1).item()
        pred_conf = pred_proba[0][pred_class].item()
    
    hablante = label_encoder.inverse_transform([pred_class])[0]
    
    return hablante, pred_conf, lemmas, attention_weights[0].cpu().numpy()[:len(sequence)]

print("\n" + "="*60)
print("EJEMPLOS DE PREDICCIÓN CON ATTENTION")
print("="*60)

ejemplos = [
    "Hoy vamos a analizar las decisiones de los entrenadores",
    "No estoy de acuerdo con eso",
    "El Real Madrid tiene que mejorar",
]

for ejemplo in ejemplos:
    resultado = predecir_hablante(
        ejemplo, model, word2vec, vocab, label_encoder, device, MAX_SEQ_LENGTH
    )
    if resultado[0]:
        hablante, confianza, lemmas, att_weights = resultado
        print(f"\nFrase: '{ejemplo}'")
        print(f"Predicción: {hablante} (confianza: {confianza:.2%})")
        print(f"Palabras más importantes (por attention):")
        
        # Mostrar top 3 palabras con mayor peso de atención
        if len(lemmas) > 0:
            top_indices = np.argsort(att_weights)[-min(3, len(lemmas)):][::-1]
            for idx in top_indices:
                print(f"  - '{lemmas[idx]}': {att_weights[idx]:.3f}")
    else:
        print(f"\nFrase: '{ejemplo}'")
        print("No se pudo predecir (sin palabras conocidas)")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
print("\nMEJORAS IMPLEMENTADAS:")
print("✓ LSTM Bidireccional (2 capas) - Captura contexto en ambas direcciones")
print("✓ Self-Attention - Pondera importancia de cada palabra")
print("✓ Layer Normalization - Estabiliza entrenamiento profundo")
print("✓ Packed Sequences - Manejo eficiente de longitudes variables")
print("✓ Gradient Clipping - Previene explosión de gradientes")
print("✓ Warmup Learning Rate - Estabiliza inicio del entrenamiento")
print("✓ Label Smoothing - Evita overconfidence, mejora generalización")
print("\nNOTA: Los embeddings de Word2Vec se han ajustado durante el entrenamiento")
print("Esto permite que el modelo adapte las representaciones al dominio específico")
