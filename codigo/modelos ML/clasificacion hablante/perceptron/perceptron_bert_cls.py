import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import ast

# CONFIGURACIÓN
SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("BERT [CLS] + PERCEPTRON")

# CARGA DE DATOS
try:
    df = pd.read_csv("dataset/dataset_bert.csv")
except:
    df = pd.read_csv("dataset/dataset_preprocesado.csv")
    df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df["text"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))

# Filtro de longitud mínima para evitar ruido
df = df[df["text"].str.len() >= 5].copy()

X = df["text"].values
y = df["speaker"].values

# Codificación etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Split Estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

# TOKENIZER & DATASET
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

# MODELO: BERT + PERCEPTRÓN OPTIMIZADO
class OptimizedBertPerceptron(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(OptimizedBertPerceptron, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        embedding_dim = self.bert.config.hidden_size
    
        self.fc1 = nn.Linear(embedding_dim, 256)
        
        # --- BLOQUE 2: 256 -> 128 ---
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        # BERT
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = output.pooler_output
        
        # Perceptrón
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Salida
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OptimizedBertPerceptron(num_classes).to(device)

# Pesos para desbalanceo
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))

# ENTRENAMIENTO

# FASE 1: BERT Congelado
print("\nFASE 1:BERT CONGELADO")
for param in model.bert.parameters():
    param.requires_grad = False

optimizer = optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(5):
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
    print(f"Epoch {epoch+1}/5 (Freeze) | Loss: {total_loss/len(train_loader):.4f}")

# FASE 2: Fine-Tuning Completo
print("\nFASE 2: BERT DESCONGELADO")
for param in model.bert.parameters():
    param.requires_grad = True

EPOCHS = 15
PATIENCE = 5
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*EPOCHS)

best_f1 = 0.0
patience_counter = 0
history = {'loss': [], 'val_f1': []}

for epoch in range(EPOCHS):
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
    history['loss'].append(avg_loss)
    
    # Validación
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, mask)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    val_acc = accuracy_score(all_targets, all_preds)
    history['val_f1'].append(val_f1)
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} (Acc: {val_acc:.4f})")
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_bert_mlp_optimized.pth')
        print(f"  Nuevo mejor modelo guardado.")
    else:
        patience_counter += 1
        print(f"  Patience {patience_counter}/{PATIENCE}")
        
    if patience_counter >= PATIENCE:
        print("Early Stopping.")
        break
    
    torch.cuda.empty_cache()

# Evaluación Final
print("\nEVALUACIÓN FINAL")
model.load_state_dict(torch.load('models/clasificacion_hablantes/best_bert_mlp_optimized.pth'))
model.eval()
final_preds, final_targets = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, mask)
        _, preds = torch.max(outputs, 1)
        final_preds.extend(preds.cpu().numpy())
        final_targets.extend(labels.cpu().numpy())

print(classification_report(final_targets, final_preds, target_names=label_encoder.classes_))

# Plots
plt.figure(figsize=(10,5))
plt.plot(history['val_f1'], label='Val F1 (Macro)')
plt.plot(history['loss'], label='Train Loss')
plt.title('Entrenamiento Optimizado BERT+MLP')
plt.legend()
plt.savefig('imagenes/training_mlp_bert_cls.png')