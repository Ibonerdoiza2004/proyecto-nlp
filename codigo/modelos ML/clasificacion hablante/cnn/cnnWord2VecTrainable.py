"""
CNN con Word2Vec Trainable
Word2Vec embeddings entrenables + CNN con múltiples kernel sizes
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
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42); torch.manual_seed(42)

EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZES = 100, 100, [2, 3, 4, 5]
DROPOUT, BATCH_SIZE, EPOCHS, MAX_LEN = 0.5, 32, 30, 100

print("CNN + WORD2VEC TRAINABLE")

df = pd.read_csv("dataset/dataset_preprocesado.csv")
df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x) if x else [])
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

texts, labels = df["lemmas_no_stop"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

print("Entrenando Word2Vec...")
w2v_model = Word2Vec(sentences=X_train, vector_size=EMBEDDING_DIM, window=5, min_count=2, workers=4, sg=1, epochs=10)

vocab = {word: idx for idx, word in enumerate(w2v_model.wv.index_to_key)}
vocab['<PAD>'] = len(vocab)
vocab['<UNK>'] = len(vocab)
vocab_size = len(vocab)

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
for word, idx in vocab.items():
    if word in w2v_model.wv:
        embedding_matrix[idx] = w2v_model.wv[word]

def tokenize(texts, vocab, max_len):
    sequences = []
    for text in texts:
        seq = [vocab.get(word, vocab['<UNK>']) for word in text[:max_len]]
        sequences.append(seq + [vocab['<PAD>']] * (max_len - len(seq)))
    return np.array(sequences)

X_train_idx = tokenize(X_train, vocab, MAX_LEN)
X_test_idx = tokenize(X_test, vocab, MAX_LEN)

class CNNWord2VecTrainable(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_sizes, num_classes, dropout, embedding_matrix):
        super(CNNWord2VecTrainable, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<PAD>'])
        self.embedding.weight.data.copy_(torch.FloatTensor(embedding_matrix))
        self.embedding.weight.requires_grad = True  # TRAINABLE
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = self.relu(bn(conv(embedded)))
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        concatenated = torch.cat(conv_outputs, dim=1)
        return self.fc(self.dropout(concatenated))

model = CNNWord2VecTrainable(vocab_size, EMBEDDING_DIM, NUM_FILTERS, KERNEL_SIZES, num_classes, DROPOUT, embedding_matrix).to(device)
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,} (Embeddings TRAINABLES)")

class_weights_tensor = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train_loader = DataLoader(list(zip(X_train_idx, y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(X_test_idx, y_test)), batch_size=BATCH_SIZE)

best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        seq = torch.LongTensor(np.array([b[0] for b in batch])).to(device)
        lbl = torch.LongTensor(np.array([b[1] for b in batch])).to(device)
        optimizer.zero_grad()
        criterion(model(seq), lbl).backward()
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            seq = torch.LongTensor(np.array([b[0] for b in batch])).to(device)
            lbl = torch.LongTensor(np.array([b[1] for b in batch])).to(device)
            correct += (torch.max(model(seq), 1)[1] == lbl).sum().item()
            total += lbl.size(0)
    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_cnn_w2v_trainable.pth')
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

model.load_state_dict(torch.load('models/best_cnn_w2v_trainable.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        seq = torch.LongTensor(np.array([b[0] for b in batch])).to(device)
        lbl = torch.LongTensor(np.array([b[1] for b in batch]))
        all_preds.extend(torch.max(model(seq), 1)[1].cpu().numpy())
        all_labels.extend(lbl.numpy())

print(f"\nTest Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('CNN + Word2Vec Trainable')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn_w2v_trainable.png', dpi=300)
print("✓ Completado")
