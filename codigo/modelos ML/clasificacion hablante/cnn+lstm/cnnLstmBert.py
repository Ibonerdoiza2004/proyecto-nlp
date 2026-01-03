import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Importación correcta para versiones modernas
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import gc

# Configuración de Semillas
SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("HYBRID: BERT (Seq) + CNN + LSTM - FINE-TUNING REAL")

# 1. PREPARACIÓN DE DATOS
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

# 2. TOKENIZER Y DATASET
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 128
BATCH_SIZE = 16 

class BERTDataset(Dataset):
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

train_dataset = BERTDataset(X_train, y_train, tokenizer, MAX_LEN)
test_dataset = BERTDataset(X_test, y_test, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. MODELO HÍBRIDO COMPLEJO (BERT -> CNN -> LSTM)
class BertCnnLstmClassifier(nn.Module):
    def __init__(self, n_classes, num_filters=64, kernel_sizes=[3], lstm_hidden=128, dropout=0.5):
        super(BertCnnLstmClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        embedding_dim = self.bert.config.hidden_size # 768
        
        # CNN: Extrae características locales de la secuencia de BERT
        # Mantenemos padding='same' para no reducir la longitud de la secuencia para la LSTM
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                      out_channels=num_filters, 
                      kernel_size=k, 
                      padding=k//2) 
            for k in kernel_sizes
        ])
        
        # LSTM: Procesa la secuencia de características extraídas por la CNN
        cnn_out_dim = num_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes) # *2 por Bidireccional
        
    def forward(self, input_ids, attention_mask):
        # 1. BERT Output
        # Shape: (Batch, Seq_Len, 768)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state 
        
        # 2. CNN (requiere Batch, Channels, Seq_Len)
        x = x.permute(0, 2, 1) 
        
        # Aplicar convoluciones y concatenar filtros
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1) # Shape: (Batch, Total_Filters, Seq_Len)
        
        # 3. Preparar para LSTM (requiere Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1)
        
        # 4. LSTM
        # output shape: (Batch, Seq_Len, Hidden*2)
        # hidden shape: (Layers*Dir, Batch, Hidden)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 5. Pooling (Usamos el último estado oculto relevante o Concatenación de fwd/bwd)
        # Tomamos el último estado oculto de ambas direcciones
        hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        x = self.dropout(hidden_final)
        logits = self.fc(x)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertCnnLstmClassifier(num_classes).to(device)

# ==========================================
# ESTRATEGIA: FREEZE -> UNFREEZE
# ==========================================

criterion = nn.CrossEntropyLoss()

# --- FASE 1: BERT CONGELADO ---
print("\n🔒 FASE 1: BERT CONGELADO (Entrenando CNN+LSTM)...")

for param in model.bert.parameters():
    param.requires_grad = False

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
EPOCHS_FREEZE = 5

for epoch in range(EPOCHS_FREEZE):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoca {epoch+1}/{EPOCHS_FREEZE} | Loss: {total_loss/len(train_loader):.4f}")

# --- FASE 2: UNFREEZE COMPLETO ---
print("\n🔓 FASE 2: FINE-TUNING COMPLETO (BERT+CNN+LSTM)...")

for param in model.bert.parameters():
    param.requires_grad = True

EPOCHS_UNFREEZE = 15
LEARNING_RATE = 2e-5 # Crítico mantener bajo
PATIENCE = 5

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS_UNFREEZE
)

best_f1 = 0.0
patience_counter = 0
history = {'train_loss': [], 'val_f1': []}

for epoch in range(EPOCHS_UNFREEZE):
    # Train
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    history['train_loss'].append(avg_loss)
    
    # Validation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, mask)
            _, preds = torch.max(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    val_acc = accuracy_score(all_targets, all_preds)
    history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS_UNFREEZE} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} (Acc: {val_acc:.4f})")
    
    # Checkpoint
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_bert_cnn_lstm.pth')
        print(f"  --> ⭐ Nuevo mejor modelo guardado (F1: {best_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  --> No mejora ({patience_counter}/{PATIENCE})")
        
    if patience_counter >= PATIENCE:
        print("Early Stopping activado.")
        break
    
    torch.cuda.empty_cache()

# 4. EVALUACIÓN FINAL
print("\n--- RESULTADOS FINALES ---")
model.load_state_dict(torch.load('models/clasificacion_hablantes/best_bert_cnn_lstm.pth'))
model.eval()

final_preds = []
final_targets = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, mask)
        _, preds = torch.max(outputs, dim=1)
        final_preds.extend(preds.cpu().numpy())
        final_targets.extend(labels.cpu().numpy())

print(classification_report(final_targets, final_preds, target_names=label_encoder.classes_, zero_division=0))

# Visualización
cm = confusion_matrix(final_targets, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'BERT + CNN + LSTM (F1: {best_f1:.2f})')
plt.savefig('imagenes/confusion_matrix_bert_cnnlstm.png')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['val_f1'], color='orange', label='Val F1')
plt.legend()
plt.savefig('imagenes/training_bert_cnnlstm.png')