import ast
import numpy as np
import pandas as pd
from gensim.models import FastText
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random

# Configuración
np.random.seed(10)
torch.manual_seed(10)
random.seed(10)
if torch.cuda.is_available():
    torch.cuda.manual_seed(10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LENGTH = 20
EMBEDDING_DIM = 100
GRU_UNITS = 256
GRU_LAYERS = 2
DROPOUT = 0.3
TEACHER_FORCING_RATIO = 0.5
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.0005

# Carga de Datos
df = pd.read_csv("dataset/dataset_preprocesado.csv")
df = df[df["speaker"] == "MIGUEL"].copy()

def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

# Carga de FastText
fasttext_model = FastText.load("models/fasttext.model")
EMBEDDING_DIM = fasttext_model.vector_size

vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
idx = 4
for word in fasttext_model.wv.index_to_key:
    vocab[word] = idx
    idx += 1

vocab_size = len(vocab)
idx_to_word = {idx: word for word, idx in vocab.items()}

def lemmas_to_indices(lemmas):
    return [vocab.get(word, vocab["<UNK>"]) for word in lemmas]

# Creación de Secuencias
sequences_X = []
sequences_y = []

for lemmas in df["lemmas_no_stop"]:
    indices = [vocab["<START>"]] + lemmas_to_indices(lemmas) + [vocab["<END>"]]
    
    for i in range(1, len(indices)):
        start_idx = max(0, i - SEQ_LENGTH)
        context = indices[start_idx:i]
        
        if len(context) < SEQ_LENGTH:
            context = [vocab["<PAD>"]] * (SEQ_LENGTH - len(context)) + context
        
        target = indices[i]
        sequences_X.append(context)
        sequences_y.append(target)

sequences_X = np.array(sequences_X)
sequences_y = np.array(sequences_y)

split_idx = int(0.9 * len(sequences_X))
X_train, X_val = sequences_X[:split_idx], sequences_X[split_idx:]
y_train, y_val = sequences_y[:split_idx], sequences_y[split_idx:]

# Matriz de Embeddings
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

embedding_matrix[vocab["<PAD>"]] = np.zeros(EMBEDDING_DIM)
for token in ["<UNK>", "<START>", "<END>"]:
    if token in vocab:
        embedding_matrix[vocab[token]] = np.random.randn(EMBEDDING_DIM) * 0.1

for word, idx in vocab.items():
    if word not in ["<PAD>", "<UNK>", "<START>", "<END>"]:
        try:
            embedding_matrix[idx] = fasttext_model.wv[word]
        except KeyError:
            embedding_matrix[idx] = np.random.randn(EMBEDDING_DIM) * 0.1

# Dataset y DataLoader
class TextGenerationDataset(Dataset):
    def __init__(self, sequences_X, sequences_y):
        self.X = torch.LongTensor(sequences_X)
        self.y = torch.LongTensor(sequences_y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextGenerationDataset(X_train, y_train)
val_dataset = TextGenerationDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Modelo GRU
class GRUGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, 
                 dropout_p=0.3, pretrained_embeddings=None, padding_idx=0):
        super(GRUGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, 
                                         padding_idx=padding_idx, _weight=pretrained_embeddings)
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_p if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x_in, hidden=None):
        embedded = self.embedding(x_in)
        gru_out, hidden = self.gru(embedded, hidden)
        last_output = gru_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return h0

# Construcción del Modelo
model = GRUGenerator(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=GRU_UNITS,
    num_layers=GRU_LAYERS,
    dropout_p=DROPOUT,
    pretrained_embeddings=embedding_matrix
).to(device)

print(model)
print(f"Parámetros totales: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer y Loss
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
)

# Entrenamiento
def train_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=0.5, epoch=0):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    current_tf_ratio = max(0.3, teacher_forcing_ratio * (0.98 ** epoch))
    
    for sequences, targets in loader:
        sequences = sequences.to(device)
        targets = targets.to(device)
        
        batch_size = sequences.size(0)
        hidden = model.init_hidden(batch_size)
        
        optimizer.zero_grad()
        use_teacher_forcing = random.random() < current_tf_ratio
        
        if use_teacher_forcing:
            outputs, _ = model(sequences, hidden)
        else:
            outputs, _ = model(sequences, hidden)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    
    return epoch_loss / len(loader), correct / total, current_tf_ratio

# Validación
def eval_epoch(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, targets in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            batch_size = sequences.size(0)
            hidden = model.init_hidden(batch_size)
            
            outputs, _ = model(sequences, hidden)
            loss = criterion(outputs, targets)
            
            epoch_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    
    return epoch_loss / len(loader), correct / total

# Loop de Entrenamiento")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(EPOCHS):
    train_loss, train_acc, tf_ratio = train_epoch(
        model, train_loader, optimizer, criterion, device, 
        teacher_forcing_ratio=TEACHER_FORCING_RATIO, epoch=epoch
    )
    val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
    
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    print(f'Epoch {epoch+1}/{EPOCHS} | TF Ratio: {tf_ratio:.3f}')
    print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}')
    print(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}')
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_gru_generator.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping en epoch {epoch+1}')
            break

# Guardado de Modelo
model.load_state_dict(torch.load('models/best_gru_generator.pth'))
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'embedding_dim': EMBEDDING_DIM,
    'hidden_dim': GRU_UNITS,
    'num_layers': GRU_LAYERS,
    'dropout': DROPOUT,
    'seq_length': SEQ_LENGTH
}, 'models/gru_text_generator.pth')

with open('models/vocab_generator.pkl', 'wb') as f:
    pickle.dump({'vocab': vocab, 'idx_to_word': idx_to_word, 'seq_length': SEQ_LENGTH}, f)

# Generación de Texto
def generate_text(model, start_text, vocab, idx_to_word, max_length=50, 
                   temperature=1.0, device=device):
    model.eval()
    context = [vocab.get(word, vocab["<UNK>"]) for word in start_text]
    generated = start_text.copy()
    
    with torch.no_grad():
        hidden = model.init_hidden(1)
        
        for _ in range(max_length):
            if len(context) > SEQ_LENGTH:
                input_seq = context[-SEQ_LENGTH:]
            else:
                input_seq = [vocab["<PAD>"]] * (SEQ_LENGTH - len(context)) + context
            
            input_tensor = torch.LongTensor([input_seq]).to(device)
            output, hidden = model(input_tensor, hidden)
            
            output = output / temperature
            probs = torch.softmax(output, dim=1)
            next_idx = torch.multinomial(probs, 1).item()
            
            if next_idx == vocab["<END>"]:
                break
            
            next_word = idx_to_word.get(next_idx, "<UNK>")
            if next_word not in ["<PAD>", "<UNK>", "<START>"]:
                generated.append(next_word)
                context.append(next_idx)
    
    return generated