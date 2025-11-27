"""
CNN con BERT CLS Token
Usa solo el embedding del token [CLS] de BERT
"""

import ast, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

BERT_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
DROPOUT, BATCH_SIZE, EPOCHS, MAX_LEN = 0.5, 16, 20, 128

print("CNN + BERT CLS TOKEN")

df = pd.read_csv("dataset/dataset_bert.csv")
df = df[df["text"].str.len() >= 10].copy()

texts, labels = df["text"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModel.from_pretrained(BERT_MODEL).to(device)
for param in bert_model.parameters():
    param.requires_grad = False

def get_bert_cls_embeddings(texts, batch_size=16):
    embeddings = []
    bert_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extrayendo CLS"):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = bert_model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings)

X_train_bert = get_bert_cls_embeddings(X_train)
X_test_bert = get_bert_cls_embeddings(X_test)
EMBEDDING_DIM = X_train_bert.shape[1]

class CNNBertCLSClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout):
        super(CNNBertCLSClassifier, self).__init__()
        # CLS es un solo vector, usamos capas fully connected
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.fc1(x))))
        x = self.dropout(self.relu(self.bn2(self.fc2(x))))
        return self.fc3(x)

model = CNNBertCLSClassifier(EMBEDDING_DIM, num_classes, DROPOUT).to(device)

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
        torch.save(model.state_dict(), 'models/best_cnn_bert_cls.pth')
    if epoch % 3 == 0:
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

model.load_state_dict(torch.load('models/best_cnn_bert_cls.pth'))
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
plt.title('CNN + BERT CLS')
plt.tight_layout()
plt.savefig('confusion_matrix_cnn_bert_cls.png', dpi=300)
print("✓ Completado")
