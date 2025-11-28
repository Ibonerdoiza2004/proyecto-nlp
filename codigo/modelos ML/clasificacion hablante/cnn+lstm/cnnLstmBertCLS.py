import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

# Configuracion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(10)
np.random.seed(10)

HIDDEN_DIM, DROPOUT, BATCH_SIZE, EPOCHS = 64, 0.4, 16, 20

print("CNN + LSTM + BERT CLS")

# Cargar datos
df = pd.read_csv("dataset/dataset_bert.csv")
df = df[df["text"].str.len() >= 10].copy()

texts, labels = df["text"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=10, stratify=labels_encoded)

# Cargar embeddings de BETO CLS
bert_cls_path = os.path.join("models", "bert_cls.npz")
bert_npz = np.load(bert_cls_path)
all_embeddings = bert_npz[bert_npz.files[0]]

# Crear un mapeo directo de texto a embedding para evitar problemas de alineación
text_to_embedding = {}
for i, text in enumerate(df["text"]):
    text_to_embedding[text] = all_embeddings[i]

# Obtener embeddings usando el mapeo directo
X_train_bert = np.array([text_to_embedding[text] for text in X_train])
X_test_bert = np.array([text_to_embedding[text] for text in X_test])
EMBEDDING_DIM = X_train_bert.shape[1]

# Definir el modelo
class CNNLSTMBertCLSClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout):
        super(CNNLSTMBertCLSClassifier, self).__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        projected = self.projection(x).unsqueeze(1)
        seq = projected.repeat(1, 5, 1)
        lstm_out, (hidden, cell) = self.lstm(seq)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

model = CNNLSTMBertCLSClassifier(EMBEDDING_DIM, HIDDEN_DIM, num_classes, DROPOUT).to(device)

# Entrenamiento
class_weights_tensor = torch.FloatTensor(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Convertir a tensores
X_train_bert = torch.FloatTensor(X_train_bert)
X_test_bert = torch.FloatTensor(X_test_bert)
y_train_tensor = torch.LongTensor(y_train)
y_test_tensor = torch.LongTensor(y_test)

# Crear datasets
from torch.utils.data import TensorDataset
train_dataset = TensorDataset(X_train_bert, y_train_tensor)
test_dataset = TensorDataset(X_test_bert, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

best_val_acc = 0
for epoch in range(EPOCHS):
    model.train()
    for emb, lbl in train_loader:
        emb, lbl = emb.to(device), lbl.to(device)
        optimizer.zero_grad()
        criterion(model(emb), lbl).backward()
        optimizer.step()
    
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for emb, lbl in test_loader:
            emb, lbl = emb.to(device), lbl.to(device)
            correct += (torch.max(model(emb), 1)[1] == lbl).sum().item()
            total += lbl.size(0)
    val_acc = correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'models/best_cnnlstm_bert_cls.pth')
    
    print(f"Epoch {epoch+1}: Val Acc = {val_acc:.4f}")

# Evaluacion
model.load_state_dict(torch.load('models/best_cnnlstm_bert_cls.pth'))
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for emb, lbl in test_loader:
        emb, lbl = emb.to(device), lbl.to(device)
        all_preds.extend(torch.max(model(emb), 1)[1].cpu().numpy())
        all_labels.extend(lbl.cpu().numpy())

print(f"\nTest Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('CNN-LSTM + BERT CLS')
plt.tight_layout()
plt.savefig('confusion_matrix_cnnlstm_bert_cls.png', dpi=300)