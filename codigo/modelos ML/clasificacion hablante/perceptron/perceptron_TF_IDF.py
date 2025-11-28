import ast

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configuración
np.random.seed(10)
torch.manual_seed(10)

print("PERCEPTRON + TF-IDF (PALABRAS)")

# CARGAR Y PREPARAR DATOS

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
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()
df["text_for_tfidf"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))

X = df["text_for_tfidf"].values
y = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)

# Vectorización con TF-IDF (palabras)
vectorizer = joblib.load('models/vec_tfidf_word.joblib')
X_train_tfidf = vectorizer.transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

num_features = X_train_tfidf.shape[1]

# CREAR DATASET
class SpeakerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {'x_data': self.X[idx], 'y_target': self.y[idx]}

train_dataset = SpeakerDataset(X_train_tfidf, y_train)
test_dataset = SpeakerDataset(X_test_tfidf, y_test)

# DEFINIR MODELO
class MLPClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features, 200)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(200, 100)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(100, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x

# CONFIGURACIÓN Y ENTRENAMIENTO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hiperparámetros
batch_size = 128
num_epochs = 150
learning_rate = 0.001

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Crear modelo
model = MLPClassifier(num_features, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
print("ENTRENAMIENTO")

train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # Modo entrenamiento
    model.train()
    running_loss = 0.0
    
    for batch in train_loader:
        x_data = batch['x_data'].to(device)
        y_target = batch['y_target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x_data)
        loss = criterion(outputs, y_target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    # Evaluación en test
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            x_data = batch['x_data'].to(device)
            y_target = batch['y_target'].to(device)
            
            outputs = model(x_data)
            _, predicted = torch.max(outputs, 1)
            total += y_target.size(0)
            correct += (predicted == y_target).sum().item()
    
    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] - Loss: {avg_loss:.4f} - Test Acc: {test_acc:.2f}%")

# EVALUACIÓN FINAL
print("EVALUACIÓN FINAL")

model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        x_data = batch['x_data'].to(device)
        y_target = batch['y_target'].to(device)
        outputs = model(x_data)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(y_target.cpu().numpy())

final_accuracy = accuracy_score(all_targets, all_preds)

print(f"Accuracy final en test: {final_accuracy:.4f}\n")
print(classification_report(all_targets, all_preds, 
                            target_names=label_encoder.classes_, 
                            zero_division=0))

# Matriz de confusión
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title('Matriz de Confusión - MLP (200-100) + TF-IDF (palabras)')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_mlp_tfidf_word.png', dpi=300, bbox_inches='tight')

# Gráfico de evolución del entrenamiento
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(range(1, num_epochs + 1), train_losses, 'b-', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Evolución del Loss durante el entrenamiento')
ax1.grid(True, alpha=0.3)

# Accuracy
ax2.plot(range(1, num_epochs + 1), test_accuracies, 'g-', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Evolución del Accuracy en Test')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_evolution_mlp_tfidf_word.png', dpi=300, bbox_inches='tight')