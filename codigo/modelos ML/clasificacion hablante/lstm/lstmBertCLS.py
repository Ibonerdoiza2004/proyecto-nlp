"""
LSTM con BERT CLS Token
Usa solo el embedding del token [CLS] en lugar del mean pooling de todos los tokens
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


HIDDEN_DIM, NUM_LAYERS, DROPOUT = 128, 2, 0.3
BATCH_SIZE, EPOCHS = 16, 15

print("LSTM + BERT CLS TOKEN")

df = pd.read_csv("dataset/dataset_bert.csv")
df = df[df["text"].str.len() >= 10].copy()

texts, labels = df["text"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)


# Cargar embeddings ya calculados de BETO CLS
import os
bert_cls_path = os.path.join("models", "bert_cls.npz")
embeddings_npz = np.load(bert_cls_path)
all_embeddings = embeddings_npz[embeddings_npz.files[0]]

# Alinear embeddings con los textos
X_train_idx = df.index[df["text"].isin(X_train)].tolist()
X_test_idx = df.index[df["text"].isin(X_test)].tolist()
X_train_bert = all_embeddings[X_train_idx]
X_test_bert = all_embeddings[X_test_idx]
EMBEDDING_DIM = X_train_bert.shape[1]

print(f"CLS embedding dim: {EMBEDDING_DIM}")

class LSTMBertCLSClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super(LSTMBertCLSClassifier, self).__init__()
        # El CLS embedding es un vector único, lo expandimos a secuencia
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x shape: (batch, embedding_dim)
        projected = self.projection(x).unsqueeze(1)  # (batch, 1, hidden_dim)
        # Repetir para crear secuencia artificial
        seq = projected.repeat(1, 5, 1)  # (batch, 5, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(seq)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

model = LSTMBertCLSClassifier(EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, num_classes, DROPOUT).to(device)

class_weights_tensor = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train_loader = DataLoader(list(zip(X_train_bert, y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(list(zip(X_test_bert, y_test)), batch_size=BATCH_SIZE)

best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        emb = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
        lbl = torch.LongTensor(np.array([b[1] for b in batch])).to(device)
        optimizer.zero_grad()
        criterion(model(emb), lbl).backward()
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            emb = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
            lbl = torch.LongTensor(np.array([b[1] for b in batch])).to(device)
            correct += (torch.max(model(emb), 1)[1] == lbl).sum().item()
            total += lbl.size(0)
    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_lstm_bert_cls.pth')
    print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

model.load_state_dict(torch.load('models/best_lstm_bert_cls.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        emb = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
        lbl = torch.LongTensor(np.array([b[1] for b in batch]))
        all_preds.extend(torch.max(model(emb), 1)[1].cpu().numpy())
        all_labels.extend(lbl.numpy())

print(f"\nTest Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('LSTM + BERT CLS Token')
plt.tight_layout()
plt.savefig('confusion_matrix_lstm_bert_cls.png', dpi=300)
print("✓ Completado")
