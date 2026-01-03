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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight # <--- FALTABA ESTO

# Configuración
np.random.seed(10)
torch.manual_seed(10)

print("PERCEPTRON (MLP) OPTIMIZADO + BAG OF WORDS (COMPARATIVA JUSTA)")

# 1. CARGAR Y PREPARAR DATOS
df = pd.read_csv("dataset/dataset_preprocesado.csv")

def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(x)
    except: return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()
df["text_for_bow"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))

X = df["text_for_bow"].values
y = df["speaker"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)

# 2. VECTORIZACIÓN (Usando vectorizador pre-existente)
print("Cargando vectorizador existente desde models/vec_bow.joblib...")
vectorizer = joblib.load('models/vec_bow.joblib')

# Transformar datos usando el vectorizador cargado
# Nota: rep_tradicional.py usa "lemmas_no_stop" unido por espacios.
# Asegurémonos de que X aquí tenga el mismo formato.
# En este script X viene de df['lemmas_no_stop'].apply(lambda x: " ".join(x)) (ver líneas anteriores si existen)
# Asumimos que X ya es una lista de strings.

X_train_bow = vectorizer.transform(X_train).toarray()
X_test_bow = vectorizer.transform(X_test).toarray()

num_features = X_train_bow.shape[1]
print(f"Dimensiones de entrada: {num_features}")

# No guardamos el vectorizador de nuevo para no sobrescribir el original
# joblib.dump(vectorizer, 'models/vec_bow.joblib') 

# 3. DATASETS
class SpeakerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return {'x_data': self.X[idx], 'y_target': self.y[idx]}

train_dataset = SpeakerDataset(X_train_bow, y_train)
test_dataset = SpeakerDataset(X_test_bow, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True) # Batch size mayor pq BoW es ligero
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 4. MODELO "ESPEJO" AL DE BERT
# Copiamos la arquitectura del script BERT Optimizado, adaptando solo la capa de entrada.
class MLPMirrorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(MLPMirrorClassifier, self).__init__()
        
        # --- BLOQUE 1: Input(5000) -> 256 ---
        # (Igualamos la neurona oculta a la de BERT)
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # <--- AÑADIDO (Igual que BERT)
        
        # --- BLOQUE 2: 256 -> 128 ---
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)  # <--- AÑADIDO (Igual que BERT)
        
        # --- BLOQUE SALIDA: 128 -> Clases ---
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activaciones y Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) # <--- SUBIDO A 0.5 (Igual que BERT)
    
    def forward(self, x):
        # Capa 1
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Capa 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Salida
        x = self.fc3(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPMirrorClassifier(num_features, num_classes).to(device)

# 5. CONFIGURACIÓN ENTRENAMIENTO (IGUALADA)
# Importante: Añadir pesos de clase para igualar condiciones con el script de BERT
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor) # <--- AÑADIDO

# Optimizador: Podemos usar Adam normal (BoW converge fácil) o AdamW. 
# Para ser puristas, usamos AdamW pero con LR más alto que BERT (BoW no necesita LR tan bajo).
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# 6. BUCLE CON EARLY STOPPING
print("\n--- INICIANDO ENTRENAMIENTO COMPARATIVO ---")
history = {'loss': [], 'val_f1': []}
best_f1 = 0.0
patience = 10
counter = 0

for epoch in range(200): # Damos muchas épocas, el early stopping cortará
    model.train()
    total_loss = 0
    for batch in train_loader:
        x = batch['x_data'].to(device)
        y_lbl = batch['y_target'].to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y_lbl)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    history['loss'].append(avg_loss)
    
    # Val
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x_data'].to(device)
            y_lbl = batch['y_target'].to(device)
            out = model(x)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            targets.extend(y_lbl.cpu().numpy())
            
    val_f1 = f1_score(targets, preds, average='macro')
    history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_bow_mlp_optimized.pth')
        print(" -> Nuevo Récord")
    else:
        counter += 1
        if counter >= patience:
            print("Early Stopping.")
            break

# Evaluación final
model.load_state_dict(torch.load('models/clasificacion_hablantes/best_bow_mlp_optimized.pth'))
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

final_f1 = f1_score(all_targets, all_preds, average='macro')

print(f"Accuracy Final: {accuracy_score(all_targets, all_preds):.4f}")
print(f"F1-Score Macro Final: {final_f1:.4f}\n")
print(classification_report(all_targets, all_preds, target_names=label_encoder.classes_, zero_division=0))

# Visualización
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(history['loss'], color=color, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('F1 Score (Macro)', color=color)
ax2.plot(history['val_f1'], color=color, label='Val F1')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Training Loss vs Validation F1 Score')
fig.tight_layout()
plt.savefig('imagenes/training_mlp_bow.png', dpi=300)

# Matriz confusión
cm = confusion_matrix(all_targets, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Matriz de Confusión - MLP BoW (F1: {final_f1:.2f})')
plt.savefig('imagenes/confusion_matrix_mlp_bow.png', dpi=300)