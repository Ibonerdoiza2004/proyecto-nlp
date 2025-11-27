"""
GRU con FastText Trainable
FastText embeddings entrenables con char n-grams (3-6) + GRU Bidireccional
"""

import ast, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from gensim.models import FastText
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42); torch.manual_seed(42)

EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT = 100, 128, 2, 0.3
BATCH_SIZE, EPOCHS, MAX_LEN = 32, 30, 100

print("GRU + FASTTEXT TRAINABLE")

df = pd.read_csv("dataset/dataset_preprocesado.csv")
df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x) if x else [])
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

texts, labels = df["lemmas_no_stop"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

print("Entrenando FastText con char n-grams...")
ft_model = FastText(sentences=X_train, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=4, 
                    sg=1, epochs=10, min_n=3, max_n=6)

vocab = {word: idx for idx, word in enumerate(ft_model.wv.index_to_key)}
vocab['<PAD>'] = len(vocab)
vocab['<UNK>'] = len(vocab)
vocab_size = len(vocab)

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, idx in vocab.items():
    if word in ft_model.wv:
        embedding_matrix[idx] = ft_model.wv[word]

def tokenize(texts, vocab, max_len):
    sequences, lengths = [], []
    for text in texts:
        seq = [vocab.get(word, vocab['<UNK>']) for word in text[:max_len]]
        sequences.append(seq + [vocab['<PAD>']] * (max_len - len(seq)))
        lengths.append(min(len(seq), max_len))
    return np.array(sequences), np.array(lengths)

X_train_idx, X_train_len = tokenize(X_train, vocab, MAX_LEN)
X_test_idx, X_test_len = tokenize(X_test, vocab, MAX_LEN)

class GRUFastTextTrainable(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout, embedding_matrix):
        super(GRUFastTextTrainable, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<PAD>'])
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding_matrix))
        self.embedding.weight.requires_grad = True  # TRAINABLE
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x, lengths):
        embedded = self.layer_norm(self.embedding(x))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.gru(packed)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

model = GRUFastTextTrainable(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, num_classes, DROPOUT, embedding_matrix).to(device)
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,} (Embeddings TRAINABLES)")

class_weights_tensor = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train_loader = DataLoader(list(zip(X_train_idx, X_train_len, y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(X_test_idx, X_test_len, y_test)), batch_size=BATCH_SIZE)

best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        seq, lengths, lbl = torch.LongTensor(np.array([b[0] for b in batch])).to(device), \
                            torch.LongTensor(np.array([b[1] for b in batch])), \
                            torch.LongTensor(np.array([b[2] for b in batch])).to(device)
        optimizer.zero_grad()
        loss = criterion(model(seq, lengths), lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            seq, lengths, lbl = torch.LongTensor(np.array([b[0] for b in batch])).to(device), \
                                torch.LongTensor(np.array([b[1] for b in batch])), \
                                torch.LongTensor(np.array([b[2] for b in batch])).to(device)
            correct += (torch.max(model(seq, lengths), 1)[1] == lbl).sum().item()
            total += lbl.size(0)
    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_gru_fasttext_trainable.pth')
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

model.load_state_dict(torch.load('models/best_gru_fasttext_trainable.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        seq, lengths, lbl = torch.LongTensor(np.array([b[0] for b in batch])).to(device), \
                            torch.LongTensor(np.array([b[1] for b in batch])), \
                            torch.LongTensor(np.array([b[2] for b in batch]))
        all_preds.extend(torch.max(model(seq, lengths), 1)[1].cpu().numpy())
        all_labels.extend(lbl.numpy())

print(f"\nTest Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('GRU + FastText Trainable')
plt.tight_layout()
plt.savefig('confusion_matrix_gru_fasttext_trainable.png', dpi=300)
print("✓ Completado")
