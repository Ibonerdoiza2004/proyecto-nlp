"""
LSTM con Bag of Words (BoW) Embeddings
BoW → Dense embedding layer → LSTM Bidireccional → Dense
"""

import ast, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(42); torch.manual_seed(42)

MAX_FEATURES = 5000
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
EPOCHS = 30
SEQ_LENGTH = 10

print("="*60)
print("LSTM + BAG OF WORDS")
print("="*60)

df = pd.read_csv("dataset/dataset_preprocesado.csv")
df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x) if x else [])
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()
df["text"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))

texts, labels = df["text"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

vectorizer = CountVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), min_df=2, max_df=0.95)
X_train_bow = vectorizer.fit_transform(X_train).toarray()
X_test_bow = vectorizer.transform(X_test).toarray()

print(f"BoW features: {X_train_bow.shape[1]}")

# Expandir a secuencias
X_train_seq = np.repeat(X_train_bow[:, np.newaxis, :], SEQ_LENGTH, axis=1)
X_test_seq = np.repeat(X_test_bow[:, np.newaxis, :], SEQ_LENGTH, axis=1)

class LSTMBoWClassifier(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super(LSTMBoWClassifier, self).__init__()
        self.embedding_layer = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.ReLU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.view(-1, input_dim)
        embedded = self.embedding_layer(x).view(batch_size, seq_len, -1)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

model = LSTMBoWClassifier(X_train_bow.shape[1], EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, num_classes, DROPOUT).to(device)
print(f"Parámetros: {sum(p.numel() for p in model.parameters()):,}")

class_weights_tensor = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train))
test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("\nEntrenando...")
best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    for seq, lbl in train_loader:
        seq, lbl = seq.to(device), lbl.to(device)
        optimizer.zero_grad()
        loss = criterion(model(seq), lbl)
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for seq, lbl in test_loader:
            seq, lbl = seq.to(device), lbl.to(device)
            pred = torch.max(model(seq), 1)[1]
            correct += (pred == lbl).sum().item()
            total += lbl.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_lstm_bow.pth')

model.load_state_dict(torch.load('models/best_lstm_bow.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for seq, lbl in test_loader:
        all_preds.extend(torch.max(model(seq.to(device)), 1)[1].cpu().numpy())
        all_labels.extend(lbl.numpy())

print("\n" + "="*60)
print("RESULTADOS")
print("="*60)
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
print(f"Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('LSTM + BoW')
plt.tight_layout()
plt.savefig('confusion_matrix_lstm_bow.png', dpi=300)
print("✓ Completado")
