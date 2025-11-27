"""
Clasificación de Hablantes usando LSTM con FastText TRAINABLE
Arquitectura: FastText embeddings (entrenables con char n-grams) → LSTM Bidireccional → Attention → Dense
Los embeddings de FastText se ajustan durante el entrenamiento
"""

import ast, numpy as np, pandas as pd
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
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

EMBEDDING_DIM = 100
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.4
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
MAX_SEQ_LENGTH = 150

print("="*60)
print("LSTM BIDIRECCIONAL + FASTTEXT TRAINABLE")
print("="*60)

print("\nCargando datos...")
df = pd.read_csv("dataset/dataset_preprocesado.csv")

def parse_list(x):
    return x if isinstance(x, list) else (ast.literal_eval(x) if x else [])

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

texts = df["lemmas_no_stop"].tolist()
labels = df["speaker"].values

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Entrenar FastText
print("\nEntrenando FastText (char n-grams 3-6)...")
ft_model = FastText(sentences=X_train, vector_size=EMBEDDING_DIM, window=5, min_count=2,
                     workers=4, sg=1, min_n=3, max_n=6, epochs=20)

vocab = {word: idx + 1 for idx, word in enumerate(ft_model.wv.index_to_key)}
vocab_size = len(vocab) + 1

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, idx in vocab.items():
    if word in ft_model.wv:
        embedding_matrix[idx] = ft_model.wv[word]

X_train_seq = [[vocab.get(w, 0) for w in text] for text in X_train]
X_test_seq = [[vocab.get(w, 0) for w in text] for text in X_test]

class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences, self.labels = sequences, labels
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return torch.LongTensor(self.sequences[idx]), self.labels[idx], len(self.sequences[idx])

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    sequences = [sequences[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]
    lengths = [min(lengths[i], MAX_SEQ_LENGTH) for i in sorted_indices]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)[:, :MAX_SEQ_LENGTH]
    return sequences_padded, torch.LongTensor(labels), torch.LongTensor(lengths)

train_loader = DataLoader(TextDataset(X_train_seq, y_train), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(TextDataset(X_test_seq, y_test), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

class LSTMFastTextTrainable(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout, pretrained_embeddings=None):
        super(LSTMFastTextTrainable, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
        self.embedding.weight.requires_grad = True  # TRAINABLE
        
        self.embed_ln = nn.LayerNorm(embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.lstm_ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, lengths):
        embedded = self.dropout(self.embed_ln(self.embedding(x)))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)
        lstm_out, (hidden, cell) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(self.lstm_ln(lstm_out))
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

model = LSTMFastTextTrainable(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, num_classes, DROPOUT, embedding_matrix).to(device)
print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

class_weights_tensor = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)

def train_epoch(model, loader):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for seq, lbl, lens in tqdm(loader, desc="Training"):
        seq, lbl, lens = seq.to(device), lbl.to(device), lens.to(device)
        optimizer.zero_grad()
        out = model(seq, lens)
        loss = criterion(out, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        loss_sum += loss.item()
        correct += (torch.max(out, 1)[1] == lbl).sum().item()
        total += lbl.size(0)
    return loss_sum / len(loader), correct / total

def eval_epoch(model, loader):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    with torch.no_grad():
        for seq, lbl, lens in tqdm(loader, desc="Evaluating"):
            seq, lbl, lens = seq.to(device), lbl.to(device), lens.to(device)
            out = model(seq, lens)
            loss_sum += criterion(out, lbl).item()
            correct += (torch.max(out, 1)[1] == lbl).sum().item()
            total += lbl.size(0)
    return loss_sum / len(loader), correct / total

print("\nEntrenando...")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(EPOCHS):
    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = eval_epoch(model, test_loader)
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.4f}, {train_acc:.4f} | Val: {val_loss:.4f}, {val_acc:.4f}")
    scheduler.step(val_loss)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_lstm_fasttext_trainable.pth')

model.load_state_dict(torch.load('models/best_lstm_fasttext_trainable.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for seq, lbl, lens in test_loader:
        seq, lens = seq.to(device), lens.to(device)
        out = model(seq, lens)
        all_preds.extend(torch.max(out, 1)[1].cpu().numpy())
        all_labels.extend(lbl.numpy())

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
print(f"Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Matriz Confusión - LSTM + FastText Trainable')
plt.tight_layout()
plt.savefig('confusion_matrix_lstm_fasttext_trainable.png', dpi=300)
print("✓ Matriz guardada")
