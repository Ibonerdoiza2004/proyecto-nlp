import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Configuración
SEED = 10
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("CNN + BERT (BETO)")

# PREPARACIÓN DE DATOS
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

# TOKENIZER Y DATASET
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

# BERT + CNN
class BertCNNClassifier(nn.Module):
    def __init__(self, n_classes, num_filters=100, kernel_sizes=[2, 3, 4], dropout=0.5):
        super(BertCNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        embedding_dim = self.bert.config.hidden_size 
        
        # CNN Layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), n_classes)
        
    def forward(self, input_ids, attention_mask):
        # BERT Output
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state 
        
        # Adaptar dimensiones para CNN
        x = x.permute(0, 2, 1) 
        
        # Convolución + Max Pooling
        conved = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        
        # Concatenación y Clasificación
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertCNNClassifier(num_classes).to(device)

criterion = nn.CrossEntropyLoss()

# FASE 1: BERT CONGELADO
print("\nFASE 1: BERT CONGELADO")

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
    print(f"  Epoch {epoch+1}/{EPOCHS_FREEZE} | Loss: {total_loss/len(train_loader):.4f}")

# TODA LA ARQUITECTURA DESCONGELADA
print("\nFASE 2: TODA LA ARQUITECTURA DESCONGELADA")

for param in model.bert.parameters():
    param.requires_grad = True

EPOCHS_UNFREEZE = 15
LEARNING_RATE = 2e-5 
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
    
    # Checkpoint & Early Stopping
    if val_f1 > best_f1:
        best_f1 = val_f1
        patience_counter = 0
        torch.save(model.state_dict(), 'models/clasificacion_hablantes/best_cnn_bert_finetuned.pth')
        print(f"  Nuevo mejor modelo guardado (F1: {best_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  No mejora ({patience_counter}/{PATIENCE})")
        
    if patience_counter >= PATIENCE:
        print("Early Stopping activado.")
        break
    
    torch.cuda.empty_cache()

# EVALUACIÓN FINAL
print("\nRESULTADOS FINALES")
model.load_state_dict(torch.load('models/clasificacion_hablantes/best_cnn_bert_finetuned.pth'))
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

# Matriz
cm = confusion_matrix(final_targets, final_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'CNN + BERT Finetuned (F1: {best_f1:.2f})')
plt.savefig('imagenes/confusion_matrix_cnn_bert.png')

# Curvas de entrenamiento
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['val_f1'], color='orange', label='Val F1')
plt.legend()
plt.savefig('imagenes/training_cnn_bert.png')