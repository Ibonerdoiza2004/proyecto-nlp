"""
CLASIFICACIÓN DE HABLANTES CON BERT + ATTENTION HEAD
Arquitectura personalizada: BERT + Mecanismo de Atención sobre la secuencia completa
"""

import ast
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
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
print("CLASIFICACIÓN CON BERT + ATTENTION HEAD")
print("=" * 70)

# Configuración
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 2e-5
PATIENCE = 5
WARMUP_RATIO = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

# 1. PREPARACIÓN DE DATOS
print("\n--- Cargando datos ---")
try:
    df = pd.read_csv("dataset/dataset_preprocesado.csv")
except FileNotFoundError:
    # Fallback por si acaso
    df = pd.read_csv("../../../dataset/dataset_preprocesado.csv")

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
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

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

# 3. ARQUITECTURA BERT + ATTENTION
print("\n--- Construyendo Modelo BERT + Attention ---")

class AttentionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Mecanismo de atención simple: W*h + b -> score
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, mask=None):
        # x: [batch, seq_len, hidden_dim]
        # mask: [batch, seq_len]
        
        # Calcular scores de atención
        scores = self.attention(x) # [batch, seq_len, 1]
        
        if mask is not None:
            # Enmascarar padding (poner valor muy bajo donde mask es 0)
            # mask es 1 para tokens reales, 0 para padding
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        
        # Softmax para obtener pesos (probabilidades)
        weights = F.softmax(scores, dim=1) # [batch, seq_len, 1]
        
        # Suma ponderada de los vectores ocultos
        # context = sum(weights * x)
        context = torch.sum(x * weights, dim=1) # [batch, hidden_dim]
        
        return context, weights

class BertAttentionClassifier(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_dim = self.bert.config.hidden_size
        
        # Capa de atención personalizada
        self.attention_head = AttentionHead(hidden_dim)
        
        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # Obtener salidas de BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Usamos last_hidden_state: [batch, seq_len, hidden_dim]
        # En lugar de pooler_output (que es solo [CLS])
        sequence_output = outputs.last_hidden_state 
        
        # Aplicar atención para obtener vector de contexto
        context_vector, attn_weights = self.attention_head(sequence_output, attention_mask)
        
        # Clasificar
        logits = self.classifier(context_vector)
        
        return logits

model = BertAttentionClassifier(MODEL_NAME, num_classes).to(device)

# Contar parámetros
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parámetros totales: {total_params:,}")

# Pesos de clase
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

# 4. ENTRENAMIENTO (2 FASES)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# --- FASE 1: BERT Congelado ---
print("\n" + "=" * 70)
print("FASE 1: Entrenar Attention Head + Clasificador (BERT Congelado)")
print("=" * 70)

# Congelar BERT
for param in model.bert.parameters():
    param.requires_grad = False

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
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
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
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(targets, predictions, average='macro')
    history_phase1['val_f1'].append(val_f1)
    print(f"Epoch {epoch+1}/5 | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")
    
    if val_f1 > best_f1_phase1:
        best_f1_phase1 = val_f1
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_bert_attention.pth')

# --- FASE 2: Fine-tuning Completo ---
print("\n" + "=" * 70)
print("FASE 2: Fine-tuning Completo (BERT Descongelado)")
print("=" * 70)

model.load_state_dict(torch.load('models/clasificacion_hablantes/best_bert_attention.pth'))

# Descongelar BERT
for param in model.bert.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*WARMUP_RATIO), num_training_steps=total_steps)

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
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
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
            logits = model(input_ids, attention_mask)
            _, preds = torch.max(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(targets, predictions, average='macro')
    val_acc = accuracy_score(targets, predictions)
    history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} (Acc: {val_acc:.4f})")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_bert_attention.pth')
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

model.load_state_dict(torch.load('models/clasificacion_hablantes/best_bert_attention.pth'))
model.eval()

final_preds = []
final_targets = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        _, preds = torch.max(logits, dim=1)
        
        final_preds.extend(preds.cpu().numpy())
        final_targets.extend(labels.cpu().numpy())

final_f1 = f1_score(final_targets, final_preds, average='macro')
final_acc = accuracy_score(final_targets, final_preds)

print(f"\nAccuracy Final: {final_acc:.4f}")
print(f"F1-Score Macro Final: {final_f1:.4f}")
print("\nReporte de Clasificación:")
print(classification_report(final_targets, final_preds, target_names=label_encoder.classes_, zero_division=0))

# 6. VISUALIZACIONES
cm = confusion_matrix(final_targets, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Matriz de Confusión - BERT + Attention (F1: {final_f1:.2f})')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('imagenes/confusion_matrix_bert_attention.png', dpi=300, bbox_inches='tight')

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
color = 'tab:orange'
ax2.set_ylabel('F1 Score (Macro)', color=color)
ax2.plot(combined_f1, color=color, label='Val F1')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('BERT + Attention Head: Training History')
fig.tight_layout()
plt.savefig('imagenes/training_history_bert_attention.png', dpi=300, bbox_inches='tight')

print("\n✅ Entrenamiento completado.")
print(f"Modelo guardado en: models/clasificacion_hablantes/best_bert_attention.pth")
