"""
CLASIFICACIÓN DE HABLANTES CON AutoModelForSequenceClassification
Enfoque genérico que permite usar cualquier transformer de HuggingFace
Modelos soportados: BETO, RoBERTa, DistilBERT, XLM-RoBERTa, etc.
"""

import ast
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# Configuración de Semillas
SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("CLASIFICACIÓN CON AutoModelForSequenceClassification")
print("=" * 70)

# Configuración - PUEDES CAMBIAR EL MODELO AQUÍ
# Opciones populares para español:
# - 'dccuchile/bert-base-spanish-wwm-cased' (BETO)
# - 'PlanTL-GOB-ES/roberta-base-bne' (RoBERTa español)
# - 'xlm-roberta-base' (Multilingual)
# - 'distilbert-base-multilingual-cased' (DistilBERT)
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-5
PATIENCE = 5
WARMUP_RATIO = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")
print(f"Modelo seleccionado: {MODEL_NAME}")

# 1. PREPARACIÓN DE DATOS
print("\n--- Cargando datos ---")
df = pd.read_csv("dataset/dataset_preprocesado.csv")

def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(x)
    except: return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()
df["text"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))

X = df["text"].values
y = df["speaker"].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"Total muestras: {len(df)}")
print(f"Clases: {list(label_encoder.classes_)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# 2. TOKENIZER Y DATASET
print("\n--- Cargando tokenizer ---")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = TextDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset = TextDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 3. MODELO
print("\n--- Cargando modelo ---")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    problem_type="single_label_classification"
)
model = model.to(device)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros totales: {total_params:,}")
print(f"Parámetros entrenables: {trainable_params:,}")

# Pesos de clase
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# 4. ENTRENAMIENTO (2 FASES)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# --- FASE 1: Backbone Congelado ---
print("\n" + "=" * 70)
print("FASE 1: Entrenar Clasificador (Backbone Congelado)")
print("=" * 70)

# Congelar el backbone del transformer (todos excepto la cabeza de clasificación)
for name, param in model.named_parameters():
    if 'classifier' not in name and 'pooler' not in name:
        param.requires_grad = False

trainable_phase1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros entrenables en Fase 1: {trainable_phase1:,}")

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

history_phase1 = {'loss': [], 'val_f1': []}
best_f1_phase1 = 0.0

for epoch in range(5):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    history_phase1['loss'].append(avg_loss)
    
    # Eval
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(targets, predictions, average='macro')
    history_phase1['val_f1'].append(val_f1)
    print(f"Epoch {epoch+1}/5 | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")
    
    if val_f1 > best_f1_phase1:
        best_f1_phase1 = val_f1
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_auto_model_classifier.pth')

# --- FASE 2: Fine-tuning Completo ---
print("\n" + "=" * 70)
print("FASE 2: Fine-tuning Completo (Backbone Descongelado)")
print("=" * 70)

model.load_state_dict(torch.load('models/clasificacion_hablantes/best_auto_model_classifier.pth'))

# Descongelar todos los parámetros
for param in model.parameters():
    param.requires_grad = True

trainable_phase2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros entrenables en Fase 2: {trainable_phase2:,}")

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * WARMUP_RATIO),
    num_training_steps=total_steps
)

history = {'loss': [], 'val_f1': []}
best_f1 = best_f1_phase1
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    history['loss'].append(avg_loss)
    
    # Eval
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(targets, predictions, average='macro')
    val_acc = accuracy_score(targets, predictions)
    history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} (Acc: {val_acc:.4f})")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_auto_model_classifier.pth')
        print(f"  --> ⭐ Nuevo mejor modelo (F1: {best_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  --> No mejora ({patience_counter}/{PATIENCE})")
    
    if patience_counter >= PATIENCE:
        print("Early stopping activado.")
        break
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 5. EVALUACIÓN FINAL
print("\n" + "=" * 70)
print("EVALUACIÓN FINAL")
print("=" * 70)

model.load_state_dict(torch.load('models/clasificacion_hablantes/best_auto_model_classifier.pth'))
model.eval()

final_preds = []
final_targets = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs.logits, dim=1)
        
        final_preds.extend(preds.cpu().numpy())
        final_targets.extend(labels.cpu().numpy())

final_f1 = f1_score(final_targets, final_preds, average='macro')
final_acc = accuracy_score(final_targets, final_preds)

print(f"\nAccuracy Final: {final_acc:.4f}")
print(f"F1-Score Macro Final: {final_f1:.4f}")
print(f"Modelo usado: {MODEL_NAME}")
print("\nReporte de Clasificación:")
print(classification_report(final_targets, final_preds, target_names=label_encoder.classes_, zero_division=0))

# 6. VISUALIZACIONES
cm = confusion_matrix(final_targets, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
model_short_name = MODEL_NAME.split('/')[-1]
plt.title(f'Matriz de Confusión - {model_short_name} (F1: {final_f1:.2f})')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('imagenes/confusion_matrix_auto_model_classifier.png', dpi=300, bbox_inches='tight')

# Combinar historiales
combined_loss = history_phase1['loss'] + history['loss']
combined_f1 = history_phase1['val_f1'] + history['val_f1']

fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color=color)
ax1.plot(combined_loss, color=color, label='Train Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label='Fase 2 inicio')

ax2 = ax1.twinx()
color = 'tab:purple'
ax2.set_ylabel('F1 Score (Macro)', color=color)
ax2.plot(combined_f1, color=color, label='Val F1')
ax2.tick_params(axis='y', labelcolor=color)

plt.title(f'AutoModel ({model_short_name}): Training History')
fig.tight_layout()
plt.savefig('imagenes/training_history_auto_model_classifier.png', dpi=300, bbox_inches='tight')

print("\n✅ Entrenamiento completado.")
print(f"Modelo guardado en: models/clasificacion_hablantes/best_auto_model_classifier.pth")
print(f"\nPara usar otro modelo, cambia MODEL_NAME en la línea 32:")
print(f"  - RoBERTa español: 'PlanTL-GOB-ES/roberta-base-bne'")
print(f"  - XLM-RoBERTa: 'xlm-roberta-base'")
print(f"  - DistilBERT: 'distilbert-base-multilingual-cased'")
