"""
CNN para clasificación de hablantes usando Word2Vec
Este modelo clasifica quién dice cada frase en el podcast usando redes convolucionales
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

MAX_SEQ_LENGTH = 150  # Longitud máxima de secuencia
EMBEDDING_DIM = 200   # Dimensión de word2vec
NUM_FILTERS = 256     # Número de filtros por kernel (aumentado)
KERNEL_SIZES = [2, 3, 4, 5]  # Tamaños de kernels (añadido kernel de tamaño 2)
DROPOUT = 0.5         # Dropout
EPOCHS = 100          # Épocas
BATCH_SIZE = 64       # Batch size
LEARNING_RATE = 0.0003 # Learning rate (reducido para entrenamiento más gradual)

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
        return torch.LongTensor(self.sequences[idx]), torch.LongTensor([self.labels[idx]])

# Función de collate para padding dinámico
def collate_fn(batch):
    sequences, labels = zip(*batch)
    # Padding de secuencias
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    # Truncar si es necesario
    if sequences_padded.size(1) > MAX_SEQ_LENGTH:
        sequences_padded = sequences_padded[:, :MAX_SEQ_LENGTH]
    # Padding a longitud fija si es menor
    if sequences_padded.size(1) < MAX_SEQ_LENGTH:
        padding = torch.zeros(sequences_padded.size(0), MAX_SEQ_LENGTH - sequences_padded.size(1), dtype=torch.long)
        sequences_padded = torch.cat([sequences_padded, padding], dim=1)
    labels = torch.cat(labels)
    return sequences_padded, labels

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

# Modelo CNN
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, output_dim, 
                 dropout_p=0.5, pretrained_embeddings=None, padding_idx=0):
        """
        CNN para clasificación de texto
        
        Args:
            vocab_size (int): número de embeddings
            embedding_dim (int): tamaño de los vectores de embedding
            num_filters (int): número de filtros por cada tamaño de kernel
            kernel_sizes (list): lista de tamaños de kernel (ej: [3, 4, 5])
            output_dim (int): número de clases
            dropout_p (float): probabilidad de dropout
            pretrained_embeddings (numpy.array): embeddings pre-entrenados (Word2Vec)
            padding_idx (int): índice que representa padding
        """
        super(CNNClassifier, self).__init__()
        
        # Capa de embedding
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
        
        # Capas convolucionales (una por cada tamaño de kernel)
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k
            )
            for k in kernel_sizes
        ])
        
        # Batch Normalization para cada convolución
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_p)
        
        # Capa fully connected con capa intermedia
        self.fc1 = nn.Linear(len(kernel_sizes) * num_filters, 128)
        self.fc2 = nn.Linear(128, output_dim)
    
    def forward(self, x_in, apply_softmax=False):
        """
        Args:
            x_in (torch.Tensor): tensor de entrada [batch_size, seq_len]
            apply_softmax (bool): aplicar softmax (False si se usa CrossEntropyLoss)
        Returns:
            prediction_vector: tensor de salida [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x_in)
        
        # Transponer para Conv1d: [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)
        
        # Aplicar cada convolución + Batch Norm + ReLU + max pooling
        conved = []
        for conv, bn in zip(self.convs, self.batch_norms):
            # Convolución: [batch_size, num_filters, seq_len - kernel_size + 1]
            conv_out = conv(embedded)
            # Batch normalization
            conv_out = bn(conv_out)
            # ReLU
            conv_out = torch.relu(conv_out)
            # Max pooling sobre toda la secuencia: [batch_size, num_filters, 1]
            pooled = torch.max(conv_out, dim=2)[0]  # [batch_size, num_filters]
            conved.append(pooled)
        
        # Concatenar todos los feature maps: [batch_size, num_filters * len(kernel_sizes)]
        cat = torch.cat(conved, dim=1)
        
        # Dropout
        cat = self.dropout(cat)
        
        # Primera capa fully connected
        hidden = torch.relu(self.fc1(cat))
        hidden = self.dropout(hidden)
        
        # Segunda capa fully connected
        prediction_vector = self.fc2(hidden)
        
        if apply_softmax:
            prediction_vector = torch.softmax(prediction_vector, dim=1)
        
        return prediction_vector

# Construcción del modelo
print("\nConstruyendo modelo CNN...")
model = CNNClassifier(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    num_filters=NUM_FILTERS,
    kernel_sizes=KERNEL_SIZES,
    output_dim=num_classes,
    dropout_p=DROPOUT,
    pretrained_embeddings=embedding_matrix
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

# Learning rate scheduler (más conservador)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
)

# Función de entrenamiento
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

# Función de evaluación
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
        torch.save(model.state_dict(), 'models/best_cnn_speaker.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping activado en epoch {epoch+1}')
            break

# Cargar mejor modelo
model.load_state_dict(torch.load('models/best_cnn_speaker.pth'))

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
plt.title('Matriz de Confusión - CNN Clasificación de Hablantes')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn.png', dpi=300, bbox_inches='tight')
print("\nMatriz de confusión guardada en: confusion_matrix_cnn.png")

# Gráficas de entrenamiento
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy
axes[0].plot(history['train_acc'], label='Train')
axes[0].plot(history['val_acc'], label='Validation')
axes[0].set_title('Accuracy durante el entrenamiento - CNN')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history['train_loss'], label='Train')
axes[1].plot(history['val_loss'], label='Validation')
axes[1].set_title('Loss durante el entrenamiento - CNN')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_cnn.png', dpi=300, bbox_inches='tight')
print("Historial de entrenamiento guardado en: training_history_cnn.png")

# Guardar modelo
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'num_filters': NUM_FILTERS,
    'kernel_sizes': KERNEL_SIZES,
    'output_dim': num_classes,
    'dropout': DROPOUT
}, 'models/cnn_speaker_classifier.pth')
print("\nModelo guardado en: models/cnn_speaker_classifier.pth")

# Guardar label encoder
import joblib
joblib.dump(label_encoder, 'models/label_encoder_speaker_cnn.joblib')
print("Label encoder guardado en: models/label_encoder_speaker_cnn.joblib")

# Guardar vocabulario
import pickle
with open('models/vocab_cnn.pkl', 'wb') as f:
    pickle.dump({'vocab': vocab, 'max_seq_length': MAX_SEQ_LENGTH}, f)
print("Vocabulario guardado en: models/vocab_cnn.pkl")

# Función de ejemplo para predecir nuevas frases
def predecir_hablante(frase, modelo, word2vec, vocab, label_encoder, device, max_seq_length=MAX_SEQ_LENGTH):
    """
    Predice el hablante de una nueva frase
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
        return None, None
    
    # Truncar o padding a longitud fija
    if len(sequence) > max_seq_length:
        sequence = sequence[:max_seq_length]
    elif len(sequence) < max_seq_length:
        sequence = sequence + [0] * (max_seq_length - len(sequence))
    
    # Convertir a tensor
    sequence_tensor = torch.LongTensor([sequence]).to(device)
    
    # Predecir
    with torch.no_grad():
        pred = modelo(sequence_tensor)
        pred_proba = torch.softmax(pred, dim=1)
        pred_class = torch.argmax(pred_proba, dim=1).item()
        pred_conf = pred_proba[0][pred_class].item()
    
    hablante = label_encoder.inverse_transform([pred_class])[0]
    
    return hablante, pred_conf

print("\n" + "="*50)
print("EJEMPLOS DE PREDICCIÓN")
print("="*50)

ejemplos = [
    "Solo uno. Solo uno. Ten el objetivo. 10 entrenadores, 10 decisiones. Vale, vale. 12 meses, 12 vidas. 12 causas. Bueno, arrancamos con Sergio Francisco, técnico de la Real Sociedad. Creo que es un tema muy interesante porque está ligado al gran rendimiento de Miquel Hoyarzabal como 9 de la selección española. Ya analizamos el otro día. El contexto es completamente diferente. Yo, por ejemplo, en el Miquel Hoyarzabal delantero centro no creo tanto en esta Real Sociedad.", #MIGUEL
    "Además de llegar como medio centro a la cantera del Cádiz y de repente que alguien le puso ahí… Llegó como medio centro. Sí, él empieza como medio centro. La temporada en la que debuta efectivamente y se empieza a salir a nivel goleador y es espectacular lo suyo.", #ALEX
    "Después de tanto tiempo sin estar en primera división, regresa al Real Oviedo. No te quedes con el Icy. Icy hubiese tenido que... Hombre, si te podías quedar con Icy Palazón...", #ADRIAN
    "Claro. Pero es tan imprescindible un jugador de buen pie. Yo creo que sobre todo lo que necesita es un buen defensor para hacer frente.", #NAHUEL
]

for ejemplo in ejemplos:
    hablante, confianza = predecir_hablante(
        ejemplo, model, word2vec, vocab, label_encoder, device, MAX_SEQ_LENGTH
    )
    if hablante:
        print(f"\nFrase: '{ejemplo}'")
        print(f"Predicción: {hablante} (confianza: {confianza:.2%})")
    else:
        print(f"\nFrase: '{ejemplo}'")
        print("No se pudo predecir (sin palabras conocidas)")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
