""""
Hemos usado IAG para generar este análisis
"""
import os
import ast
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gensim.models import FastText, Word2Vec
from transformers import BertModel, BertTokenizer, AutoModelForSequenceClassification
import pickle
import joblib

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use('default')

MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased'
MAX_LEN = 128

print("ANÁLISIS COMPARATIVO COMPLETO DE TODOS LOS MODELOS DE DEEP LEARNING")
print("=" * 75)
print("Modelos evaluados:")
print("  - BERT Fine-tuned: 7 modelos (incluye Attention y AutoModel)")
print("  - FastText: 4 modelos")
print("  - Word2Vec: 4 modelos")
print("  - Perceptrones (PyTorch): 3 modelos")
print("  - Shallow Learning (Sklearn): 4 modelos")
print("  TOTAL: 22 modelos")
print("=" * 75)

# Cargar datos base
print("Cargando datos...")
df = pd.read_csv("dataset/dataset_preprocesado.csv")

def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

# --- NUEVO: Preparar datos para BERT DL (Joined Lemmas) ---
# Esto coincide con cnnBert.py, lstmBert.py, etc.
df_bert_dl = df.copy()
df_bert_dl["text"] = df_bert_dl["lemmas_no_stop"].apply(lambda x: " ".join(x))
texts_bert_dl = df_bert_dl["text"].tolist()
labels_bert_dl = df_bert_dl["speaker"].values

# Preparar datos para BERT (Shallow / Original)
df_bert = pd.read_csv("dataset/dataset_bert.csv")

# Parse embeddings from CSV to avoid alignment issues with .npz
import ast
def parse_embedding(x):
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x
    except:
        return []

print("Cargando embeddings BERT desde CSV...")
df_bert['bert_cls'] = df_bert['bert_cls'].apply(parse_embedding)

df_bert = df_bert[df_bert["text"].str.len() >= 5].copy()  # Mismo filtro que en entrenamiento
texts_bert = df_bert["text"].tolist()
labels_bert = df_bert["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_bert)
num_classes = len(label_encoder.classes_)

# Codificar etiquetas para BERT DL
labels_encoded_dl = label_encoder.transform(labels_bert_dl)

# Split para BERT (Shallow)
X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
    texts_bert, labels_encoded, test_size=0.2, random_state=10, stratify=labels_encoded
)

# Split para BERT DL
X_train_bert_dl, X_test_bert_dl, y_train_bert_dl, y_test_bert_dl = train_test_split(
    texts_bert_dl, labels_encoded_dl, test_size=0.2, random_state=10, stratify=labels_encoded_dl
)

# Preparar datos para modelos de texto (FastText, Word2Vec)
texts = df["lemmas_no_stop"].tolist()
labels = df["speaker"].values
y_encoded = label_encoder.fit_transform(labels)

X_train_texts, X_test_texts, y_train_texts, y_test_texts = train_test_split(
    texts, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)

# Preparar textos para perceptrones (como strings)
df["text_for_perceptrons"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))
texts_for_perceptrons = df["text_for_perceptrons"].tolist()
X_train_texts_bow, X_test_texts_bow, _, _ = train_test_split(
    texts_for_perceptrons, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)

print(f"Dataset BERT: {len(df_bert)} muestras")
print(f"Dataset Texto: {len(df)} muestras")
print(f"Clases: {list(label_encoder.classes_)}")
print(f"Distribución de clases en test BERT:")
for i, clase in enumerate(label_encoder.classes_):
    count = sum(y_test_bert == i)
    print(f"  {clase}: {count} muestras ({count/len(y_test_bert)*100:.1f}%)")
print()

# Crear directorio para gráficos
os.makedirs('imagenes/analisis/clasificacion_hablantes', exist_ok=True)

# Definir todas las clases de modelos
import torch.nn as nn

# --- MODELOS BERT FINE-TUNED ---

class OptimizedBertPerceptron(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(OptimizedBertPerceptron, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        embedding_dim = self.bert.config.hidden_size # 768
        
        # --- BLOQUE 1: 768 -> 256 ---
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.ln1 = nn.LayerNorm(256)  # Mejora clave: Normalización
        
        # --- BLOQUE 2: 256 -> 128 ---
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)  # Mejora clave: Normalización
        
        # --- BLOQUE SALIDA: 128 -> Clases ---
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activaciones y Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) # Mejora clave: Dropout 0.5
        
    def forward(self, input_ids, attention_mask):
        # 1. BERT
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = output.pooler_output # Token [CLS]
        
        # 2. Perceptrón Optimizado
        # Capa 1
        x = self.fc1(x)
        x = self.ln1(x)     # Normalizar
        x = self.relu(x)    # Activar
        x = self.dropout(x) # Regularizar
        
        # Capa 2
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Salida
        x = self.fc3(x)
        return x

class BertCNNClassifier(nn.Module):
    def __init__(self, n_classes, num_filters=128, kernel_sizes=[2, 3, 4], dropout=0.3):
        super(BertCNNClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        embedding_dim = self.bert.config.hidden_size 
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), n_classes)
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state 
        x = x.permute(0, 2, 1) 
        conved = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        cat = torch.cat(pooled, dim=1)
        cat = self.dropout(cat)
        return self.fc(cat)

class BertLstmClassifier(nn.Module):
    def __init__(self, n_classes, lstm_hidden=128, num_layers=2, dropout=0.3):
        super(BertLstmClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        embedding_dim = self.bert.config.hidden_size
        self.lstm_hidden = lstm_hidden  # Guardar como atributo
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state
        _, (hidden, cell) = self.lstm(sequence_output)
        hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.dropout(hidden_final)
        return self.fc(x)

class BertGruClassifier(nn.Module):
    def __init__(self, n_classes, hidden_dim=128, num_layers=1, dropout=0.3):
        super(BertGruClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        embedding_dim = self.bert.config.hidden_size
        self.hidden_dim = hidden_dim  # Guardar como atributo
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_out.last_hidden_state
        _, hidden = self.gru(sequence_output)
        hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.dropout(hidden_final)
        return self.fc(x)

class BertCnnLstmClassifier(nn.Module):
    def __init__(self, n_classes, num_filters=64, kernel_sizes=[2, 3, 4], lstm_hidden=64, dropout=0.3):
        super(BertCnnLstmClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
        embedding_dim = self.bert.config.hidden_size
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k, padding=k//2) 
            for k in kernel_sizes
        ])
        
        cnn_out_dim = num_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = bert_out.last_hidden_state 
        x = x.permute(0, 2, 1) 
        x = [torch.relu(conv(x)) for conv in self.convs]
        x = torch.cat(x, dim=1) 
        x = x.permute(0, 2, 1)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        x = self.dropout(hidden_final)
        return self.fc(x)

# --- MODELOS PERCEPTRON (BERT CLS) ---
# --- MODELOS FASTTEXT / WORD2VEC ---

class CNNClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters, kernel_sizes, num_classes, dropout):
        super(CNNClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True 
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(embedded)
            conv_out = bn(conv_out)
            conv_out = torch.relu(conv_out)
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        concatenated = torch.cat(conv_outputs, dim=1)
        out = self.dropout(concatenated)
        logits = self.fc(out)
        return logits

class CNNClassifier_Word2Vec(nn.Module):
    def __init__(self, embedding_matrix, num_filters, kernel_sizes, num_classes, dropout):
        super(CNNClassifier_Word2Vec, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True 
        
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Architecture from cnnWord2Vec.py (2 layers FC)
        self.fc1 = nn.Linear(num_filters * len(kernel_sizes), 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(embedded)
            conv_out = bn(conv_out)
            conv_out = torch.relu(conv_out)
            pooled = torch.max(conv_out, dim=2)[0]
            conv_outputs.append(pooled)
        
        concatenated = torch.cat(conv_outputs, dim=1)
        
        # Dropout before fc1
        out = self.dropout(concatenated)
        
        # fc1 -> relu -> dropout
        hidden = torch.relu(self.fc1(out))
        hidden = self.dropout(hidden)
        
        # fc2
        logits = self.fc2(hidden)
        return logits

class CNNLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters, kernel_sizes, hidden_dim, lstm_layers, num_classes, dropout):
        super(CNNLSTMClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True 
        
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k)
            for k in kernel_sizes
        ])
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in kernel_sizes
        ])
        
        self.lstm = nn.LSTM(
            input_size=num_filters * len(kernel_sizes),
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.transpose(1, 2)
        
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_out = conv(embedded)
            conv_out = bn(conv_out)
            conv_out = self.relu(conv_out)
            conv_outputs.append(conv_out)
        
        max_len = max(out.size(2) for out in conv_outputs)
        padded_outputs = []
        for out in conv_outputs:
            if out.size(2) < max_len:
                padding_size = max_len - out.size(2)
                padded = torch.nn.functional.pad(out, (0, padding_size), mode='constant', value=0)
                padded_outputs.append(padded)
            else:
                padded_outputs.append(out)
        
        cnn_features = torch.cat(padded_outputs, dim=1)
        cnn_features = cnn_features.transpose(1, 2)
        cnn_features = self.dropout(cnn_features)
        
        lstm_out, (hidden, cell) = self.lstm(cnn_features)
        hidden_final = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden_final)

class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(BiGRUClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True 
        
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded)
        hidden_fwd = hidden[-2, :, :]
        hidden_bwd = hidden[-1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        hidden_concat = self.dropout(hidden_concat)
        return self.fc(hidden_concat)

class BiGRUClassifier_Word2Vec(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(BiGRUClassifier_Word2Vec, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True 
        
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, hidden = self.gru(embedded)
        # Word2Vec training script uses indices 0 and 1 (first layer) instead of -2 and -1
        hidden_fwd = hidden[0, :, :]
        hidden_bwd = hidden[1, :, :]
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        hidden_concat = self.dropout(hidden_concat)
        return self.fc(hidden_concat)

class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super(BiLSTMClassifier, self).__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True 
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        output, (hidden, cell) = self.lstm(embedded)
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        final_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        out = self.dropout(final_hidden)
        return self.fc(out)

# --- MODELOS PERCEPTRON (BOW/TF-IDF) ---
class MLPMirrorClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(MLPMirrorClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc3(x)

# LSTM Speaker Classifier (2 layers, 256 hidden, embeddings frozen)
class LSTMSpeakerClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes, dropout):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, 256, num_layers=2, batch_first=True,
                           dropout=dropout if 2 > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# CNN Speaker Classifier (4 conv layers, 256 filters each)
class CNNSpeakerClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_classes, dropout):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, 256, kernel_size=k)
            for k in [2, 3, 4, 5]
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(256)
            for _ in [2, 3, 4, 5]
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_results = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_result = self.relu(bn(conv(x)))
            pooled = torch.max(conv_result, dim=2)[0]
            conv_results.append(pooled)
        x = torch.cat(conv_results, dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# BERT LSTM Speaker Classifier (2 layers, 256 hidden, attention)
class BERTLSTMSpeakerClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, 256, num_layers=2, batch_first=True,
                           dropout=dropout if 2 > 1 else 0, bidirectional=True)

        self.dropout = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = nn.ModuleDict({
            'Wa': nn.Linear(512, 512),  # 512 = 256*2
            'Ua': nn.Linear(512, 512),
            'Va': nn.Linear(512, 1)
        })

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, embedding_dim]
        lstm_out, (hidden, cell) = self.lstm(x)  # [batch, 1, 512]

        # Attention
        Wa_h = self.attention['Wa'](lstm_out)  # [batch, 1, 512]
        Ua_h = self.attention['Ua'](lstm_out)  # [batch, 1, 512]
        tanh_sum = torch.tanh(Wa_h + Ua_h)  # [batch, 1, 512]
        e = self.attention['Va'](tanh_sum).squeeze(-1)  # [batch, 1]
        alpha = torch.softmax(e, dim=1).unsqueeze(-1)  # [batch, 1, 1]
        context = torch.sum(alpha * lstm_out, dim=1)  # [batch, 512]

        return self.fc(self.dropout(context))

# CNN-LSTM + BERT (mean pooling)
class CNNLSTMBERTClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(128 * (embedding_dim // 4), hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1).unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# LSTM + BERT CLS
class LSTMBertCLSClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        projected = self.projection(x).unsqueeze(1)
        seq = projected.repeat(1, 5, 1)
        lstm_out, (hidden, cell) = self.lstm(seq)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# LSTM + BERT (mean pooling)
class BERTLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, embedding_dim]
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# GRU + BERT CLS
class GRUBertCLSClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        projected = self.projection(x).unsqueeze(1)
        seq = projected.repeat(1, 5, 1)
        gru_out, hidden = self.gru(seq)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# GRU + BERT (mean pooling)
class GRUEmbeddingsClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        gru_out, hidden = self.gru(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# CNN + FastText/Word2Vec (checkpoint dimensions - 128 filters)
class CNNClassifier_Checkpoint(nn.Module):
    def __init__(self, embedding_matrix, num_classes, dropout):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, 128, kernel_size=k)  # 128 filters
            for k in [2, 3, 4, 5]
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(128)  # 128
            for _ in [2, 3, 4, 5]
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 4, num_classes)  # 512

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_results = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_result = torch.relu(bn(conv(x)))
            pooled = torch.max(conv_result, dim=2)[0]
            conv_results.append(pooled)
        x = torch.cat(conv_results, dim=1)
        return self.fc(self.dropout(x))



# --- MODELOS TRANSFORMERS AVANZADOS ---

class AttentionHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # Mecanismo de atención simple: W*h + b -> score
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, hidden_states, attention_mask):
        # hidden_states: [batch, seq_len, hidden_dim]
        # attention_mask: [batch, seq_len]
        
        # Calcular scores de atención
        scores = self.attention(hidden_states)  # [batch, seq_len, 1]
        scores = scores.squeeze(-1)  # [batch, seq_len]
        
        # Aplicar máscara (poner -inf donde mask=0)
        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax para obtener pesos
        attn_weights = torch.softmax(scores, dim=1)  # [batch, seq_len]
        
        # Aplicar pesos a hidden states
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * hidden_states, dim=1)  # [batch, hidden_dim]
        
        return context_vector, attn_weights

class BertAttentionClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(MODEL_NAME)
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
        sequence_output = outputs.last_hidden_state 
        
        # Aplicar atención para obtener vector de contexto
        context_vector, attn_weights = self.attention_head(sequence_output, attention_mask)
        
        # Clasificar
        logits = self.classifier(context_vector)
        
        return logits

# Funciones auxiliares para cargar embeddings
def load_bert_embeddings(embeddings_type='cls'):
    if embeddings_type == 'cls':
        path = "models/bert_cls.npz"
    elif embeddings_type == 'mean':
        path = "models/bert_mean.npz"
    else:
        raise ValueError("Tipo de embeddings debe ser 'cls' o 'mean'")

    embeddings_npz = np.load(path)
    all_embeddings = embeddings_npz[embeddings_npz.files[0]]

    text_to_embedding = {}
    for i, text in enumerate(df_bert["text"]):
        text_to_embedding[text] = all_embeddings[i]

    X_test_embeddings = np.array([text_to_embedding[text] for text in X_test_bert])
    return torch.FloatTensor(X_test_embeddings).to(device)

def create_text_vocab_and_embeddings(model_type):
    # Cargar vocabulario común generado en word2vec.py
    print("Cargando vocabulario común desde models/word2idx.pkl...")
    with open("models/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    
    vocab_size = len(word2idx)
    max_length = max(len(text) for text in texts)

    if model_type == 'fasttext':
        model = FastText.load('models/fasttext.model')
        embedding_dim = model.vector_size
    elif model_type == 'word2vec':
        model = Word2Vec.load('models/w2v.model')
        embedding_dim = model.vector_size
    else:
        raise ValueError("Tipo de modelo debe ser 'fasttext' o 'word2vec'")

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        if word in ['<pad>', '<unk>']:
            continue
        if word in model.wv:
            embedding_matrix[idx] = model.wv[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return word2idx, embedding_matrix, max_length

def create_text_vocab_and_embeddings_fixed(model_type, target_vocab_size, embedding_dim):
    """Crea vocab y embeddings con tamaño fijo para coincidir con checkpoints"""
    all_words = [word for text in texts for word in text]
    # Cargar vocabulario común
    print("Cargando vocabulario común desde models/word2idx.pkl...")
    with open("models/word2idx.pkl", "rb") as f:
        word2idx = pickle.load(f)
    
    # Si el vocabulario guardado es diferente al target, ajustamos (aunque idealmente deberían coincidir)
    current_size = len(word2idx)
    if current_size < target_vocab_size:
        for i in range(current_size, target_vocab_size):
            word2idx[f'<extra_{i}>'] = i
    
    max_length = max(len(text) for text in texts)

    # Cargar modelo de embeddings
    if model_type == 'fasttext':
        model = FastText.load('models/fasttext.model')
    elif model_type == 'word2vec':
        model = Word2Vec.load('models/w2v.model')
    else:
        raise ValueError("Tipo de modelo debe ser 'fasttext' o 'word2vec'")

    # Crear embedding matrix
    embedding_matrix = np.zeros((target_vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        if idx >= target_vocab_size: continue # Ignorar si excede
        
        if word in ['<pad>', '<unk>'] or str(word).startswith('<extra_'):
            continue
            
        if word in model.wv:
            embedding_matrix[idx] = model.wv[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    return word2idx, embedding_matrix, max_length

class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        indices = [self.word2idx.get(word, 1) for word in tokens]

        if len(indices) < self.max_length:
            indices = indices + [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# --- DATASETS ---

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

class SpeakerDataset(Dataset):
    def __init__(self, texts, labels, word2idx, max_length):
        self.texts = texts
        self.labels = labels
        self.word2idx = word2idx
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        indices = [self.word2idx.get(word, 1) for word in tokens] # 1 is <unk>
        
        if len(indices) < self.max_length:
            indices = indices + [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Modelos a evaluar - TODOS los modelos disponibles
model_configs = {
    # ========================================
    # BERT FINE-TUNED (7 modelos)
    # ========================================
    'BERT - Perceptron Optimizado': {
        'path': 'models/clasificacion_hablantes/best_bert_mlp_optimized.pth',
        'class': OptimizedBertPerceptron,
        'args': {'num_classes': 5, 'dropout': 0.5},
        'type': 'bert_finetuned'
    },
    'BERT - CNN': {
        'path': 'models/clasificacion_hablantes/best_cnn_bert_finetuned.pth',
        'class': BertCNNClassifier,
        'args': {'n_classes': 5, 'num_filters': 100, 'kernel_sizes': [2, 3, 4], 'dropout': 0.5},
        'type': 'bert_finetuned'
    },
    'BERT - CNN-LSTM': {
        'path': 'models/clasificacion_hablantes/best_bert_cnn_lstm.pth',
        'class': BertCnnLstmClassifier,
        'args': {'n_classes': 5, 'num_filters': 64, 'kernel_sizes': [3], 'lstm_hidden': 128, 'dropout': 0.3},
        'type': 'bert_finetuned'
    },
    'BERT - LSTM': {
        'path': 'models/clasificacion_hablantes/best_bert_lstm.pth',
        'class': BertLstmClassifier,
        'args': {'n_classes': 5, 'lstm_hidden': 128, 'num_layers': 2, 'dropout': 0.3},
        'type': 'bert_finetuned'
    },
    'BERT - GRU': {
        'path': 'models/clasificacion_hablantes/best_bert_gru.pth',
        'class': BertGruClassifier,
        'args': {'n_classes': 5, 'hidden_dim': 128, 'num_layers': 1, 'dropout': 0.3},
        'type': 'bert_finetuned'
    },
    'BERT - Attention': {
        'path': 'models/clasificacion_hablantes/best_bert_attention.pth',
        'class': BertAttentionClassifier,
        'args': {'num_classes': 5, 'dropout': 0.3},
        'type': 'bert_finetuned'
    },
    'BERT - AutoModel': {
        'path': 'models/clasificacion_hablantes/best_auto_model_classifier.pth',
        'class': None,  # Usa AutoModelForSequenceClassification directamente
        'args': {},
        'type': 'bert_automodel'
    },

    # ========================================
    # FASTTEXT (4 modelos)
    # ========================================
    'FastText - CNN': {
        'path': 'models/clasificacion_hablantes/best_cnn_fasttext.pth',
        'class': CNNClassifier,
        'args': {'num_filters': 128, 'kernel_sizes': [2, 3, 4, 5], 'num_classes': 5, 'dropout': 0.5},
        'type': 'fasttext'
    },
    'FastText - CNN-LSTM': {
        'path': 'models/clasificacion_hablantes/best_cnnlstm_fasttext.pth',
        'class': CNNLSTMClassifier,
        'args': {'num_filters': 64, 'kernel_sizes': [2, 3, 4], 'hidden_dim': 128, 'lstm_layers': 1, 'num_classes': 5, 'dropout': 0.5},
        'type': 'fasttext'
    },
    'FastText - BiLSTM': {
        'path': 'models/clasificacion_hablantes/best_bilstm_fasttext.pth',
        'class': BiLSTMClassifier,
        'args': {'hidden_dim': 128, 'num_layers': 2, 'num_classes': 5, 'dropout': 0.3},
        'type': 'fasttext'
    },
    'FastText - BiGRU': {
        'path': 'models/clasificacion_hablantes/best_gru_fasttext.pth',
        'class': BiGRUClassifier,
        'args': {'hidden_dim': 128, 'num_layers': 2, 'num_classes': 5, 'dropout': 0.3},
        'type': 'fasttext'
    },

    # ========================================
    # WORD2VEC (4 modelos)
    # ========================================
    'Word2Vec - CNN': {
        'path': 'models/clasificacion_hablantes/best_cnn_word2vec.pth',
        'class': CNNClassifier_Word2Vec,
        'args': {'num_filters': 256, 'kernel_sizes': [2, 3, 4, 5], 'num_classes': 5, 'dropout': 0.5},
        'type': 'word2vec'
    },
    'Word2Vec - CNN-LSTM': {
        'path': 'models/clasificacion_hablantes/best_cnnlstm_w2v.pth',
        'class': CNNLSTMClassifier,
        'args': {'num_filters': 64, 'kernel_sizes': [2, 3, 4], 'hidden_dim': 128, 'lstm_layers': 1, 'num_classes': 5, 'dropout': 0.5},
        'type': 'word2vec'
    },
    'Word2Vec - BiLSTM': {
        'path': 'models/clasificacion_hablantes/best_lstm_w2v.pth',
        'class': BiLSTMClassifier,
        'args': {'hidden_dim': 256, 'num_layers': 2, 'num_classes': 5, 'dropout': 0.3},
        'type': 'word2vec'
    },
    'Word2Vec - BiGRU': {
        'path': 'models/clasificacion_hablantes/best_bigru_w2v.pth',
        'class': BiGRUClassifier_Word2Vec,
        'args': {'hidden_dim': 128, 'num_layers': 2, 'num_classes': 5, 'dropout': 0.3},
        'type': 'word2vec'
    },

    # ========================================
    # PERCEPTRONES (PyTorch - 3 modelos)
    # ========================================
    'Perceptron - BoW': {
        'path': 'models/clasificacion_hablantes/best_bow_mlp_optimized.pth',
        'class': MLPMirrorClassifier,
        'args': {'input_dim': 4964, 'num_classes': 5, 'dropout': 0.5},
        'type': 'bow_mlp'
    },
    'Perceptron - TF-IDF Word': {
        'path': 'models/clasificacion_hablantes/best_tfidf_mlp_optimized.pth',
        'class': MLPMirrorClassifier,
        'args': {'input_dim': 5000, 'num_classes': 5, 'dropout': 0.5},
        'type': 'tfidf_mlp'
    },
    'Perceptron - TF-IDF Char': {
        'path': 'models/clasificacion_hablantes/best_tfidf_char_mlp_optimized.pth',
        'class': MLPMirrorClassifier,
        'args': {'input_dim': 5000, 'num_classes': 5, 'dropout': 0.5},
        'type': 'tfidf_char_mlp'
    },

    # ========================================
    # SHALLOW LEARNING (Sklearn - 4 modelos)
    # ========================================
    'Shallow - BoW': {
        'path': 'models/clasificacion_hablantes/best_shallow_bow.joblib',
        'vectorizer': 'models/vec_bow.joblib',
        'type': 'sklearn_shallow'
    },
    'Shallow - TF-IDF Word': {
        'path': 'models/clasificacion_hablantes/best_shallow_tfidf.joblib',
        'vectorizer': 'models/vec_tfidf_word.joblib',
        'type': 'sklearn_shallow'
    },
    'Shallow - TF-IDF Char': {
        'path': 'models/clasificacion_hablantes/best_shallow_tfidf_char.joblib',
        'vectorizer': 'models/vec_tfidf_char.joblib',
        'type': 'sklearn_shallow'
    },
    'Shallow - BERT CLS': {
        'path': 'models/clasificacion_hablantes/best_shallow_bert_cls.joblib',
        'vectorizer': None,  # Uses pre-computed BERT CLS embeddings
        'type': 'sklearn_bert_cls'
    }
}

def load_model_weights(model, path):
    """Carga pesos de modelo manejando diferentes formatos de guardado"""
    try:
        if path.endswith('.joblib'):
            # Modelos PyTorch guardados con joblib
            loaded_model = joblib.load(path)
            model.load_state_dict(loaded_model.state_dict())
            return True
        else:
            # Modelos PyTorch guardados con torch.save
            checkpoint = torch.load(path, map_location=device, weights_only=False)

            # Si tiene 'model_state_dict', usar ese
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Intentar cargar los pesos
            model.load_state_dict(state_dict, strict=False)
            return True

    except Exception as e:
        print(f"  ❌ Error cargando pesos: {e}")
        return False

results = {}

print("Evaluando modelos...")
print("-" * 50)

for model_name, config in model_configs.items():
    try:
        print(f"Evaluando: {model_name}")

        if not os.path.exists(config['path']):
            print(f"  ❌ Modelo {config['path']} no encontrado")
            continue

        model_type = config['type']
        model_class = config.get('class') # Use .get() as sklearn models don't have 'class'
        model_args = config.get('args', {}) # Use .get() as sklearn models don't have 'args'

        # 1. Prepare Data & Model
        if model_type == 'bert_finetuned':
            # BERT Fine-tuned: Raw text -> Tokenizer -> Model
            tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
            
            # FIX: BERT - Perceptron Optimizado was trained on Raw Text (dataset_bert.csv)
            # Others (CNN, LSTM) were trained on Joined Lemmas (dataset_preprocesado.csv)
            if model_name == 'BERT - Perceptron Optimizado':
                 test_dataset = BERTDataset(X_test_bert, y_test_bert, tokenizer, MAX_LEN)
            else:
                 # USAR DATASET DL (Joined Lemmas)
                 test_dataset = BERTDataset(X_test_bert_dl, y_test_bert_dl, tokenizer, MAX_LEN)

            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            model = model_class(**model_args).to(device)

        elif model_type == 'bert_automodel':
            # BERT AutoModel: Uses AutoModelForSequenceClassification
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            # USAR DATASET DL (Joined Lemmas)
            test_dataset = BERTDataset(X_test_bert_dl, y_test_bert_dl, tokenizer, MAX_LEN)
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
            # Create model using AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, 
                num_labels=num_classes,
                ignore_mismatched_sizes=True
            ).to(device)

        elif model_type in ['fasttext', 'word2vec']:
            # FastText/Word2Vec: Raw text -> Vocab -> Indices -> Model
            # We need to create embedding matrix first to pass to model init
            word2idx, embedding_matrix, max_length = create_text_vocab_and_embeddings(model_type)
            
            test_dataset = SpeakerDataset(X_test_texts, y_test_texts, word2idx, max_length)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Add embedding_matrix to args
            model_args_copy = model_args.copy()
            model_args_copy['embedding_matrix'] = embedding_matrix
            model = model_class(**model_args_copy).to(device)

        elif model_type in ['bow_mlp', 'tfidf_mlp', 'tfidf_char_mlp']:
            # Shallow MLP: Raw text -> Vectorizer -> Sparse Vector -> Model
            if model_type == 'bow_mlp':
                vectorizer = joblib.load('models/vec_bow.joblib')
            elif model_type == 'tfidf_mlp':
                vectorizer = joblib.load('models/vec_tfidf_word.joblib')
            else:
                vectorizer = joblib.load('models/vec_tfidf_char.joblib')
                
            X_test_transformed = vectorizer.transform(X_test_texts_bow).toarray()
            
            # Usar la dimensión real del vectorizador
            actual_input_dim = X_test_transformed.shape[1]
            model_args_copy = model_args.copy()
            model_args_copy['input_dim'] = actual_input_dim
            
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_transformed), 
                torch.LongTensor(y_test_texts)
            )
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            model = model_class(**model_args_copy).to(device)

        elif model_type == 'sklearn_shallow':
            # Sklearn Models: Raw text -> Vectorizer -> Model.predict
            vectorizer = joblib.load(config['vectorizer'])
            X_test_transformed = vectorizer.transform(X_test_texts_bow)
            
            model = joblib.load(config['path'])
            
            # Sklearn models don't use DataLoader or GPU usually
            all_preds = model.predict(X_test_transformed)
            all_targets = y_test_texts
            
            # Skip the PyTorch evaluation loop
            f1 = f1_score(all_targets, all_preds, average='macro')
            
            arch_name = 'Shallow Learning'
            emb_name = model_name.split(' - ')[1] if ' - ' in model_name else 'Unknown'

            results[model_name] = {
                'f1_score': f1,
                'architecture': arch_name,
                'embedding': emb_name
            }
            print(f"  ✅ F1 Score (Macro): {f1:.4f}")
            continue

        elif model_type == 'sklearn_bert_cls':
            # Sklearn Models with BERT CLS embeddings: Pre-computed embeddings -> Model.predict
            # Use embeddings directly from the dataframe (already aligned)
            
            # Create a map from text to embedding
            # Note: df_bert is already loaded and parsed at the top
            text_to_embedding = {text: emb for text, emb in zip(df_bert["text"], df_bert["bert_cls"])}
            
            # Retrieve embeddings for the test set
            X_test_embeddings = []
            valid_indices = []
            for i, text in enumerate(X_test_bert):
                if text in text_to_embedding:
                    X_test_embeddings.append(text_to_embedding[text])
                    valid_indices.append(i)
                else:
                    # Should not happen if X_test_bert comes from df_bert, but just in case
                    pass
            
            X_test_embeddings = np.array(X_test_embeddings)
            y_test_bert_valid = y_test_bert[valid_indices]
            
            model = joblib.load(config['path'])
            
            # Sklearn models don't use DataLoader or GPU
            all_preds = model.predict(X_test_embeddings)
            all_targets = y_test_bert_valid
            
            # Skip the PyTorch evaluation loop
            f1 = f1_score(all_targets, all_preds, average='macro')
            
            arch_name = 'Shallow Learning'
            emb_name = 'BERT CLS'

            results[model_name] = {
                'f1_score': f1,
                'architecture': arch_name,
                'embedding': emb_name
            }
            print(f"  ✅ F1 Score (Macro): {f1:.4f}")
            continue

        else:
            print(f"  ⚠️  Tipo {model_type} no soportado")
            continue

        # 2. Load Weights
        if not load_model_weights(model, config['path']):
            continue

        # 3. Evaluate
        model.eval()
        all_preds = []
        all_targets = [] # To be safe, though we have y_test_*
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, dict): # BERTDataset returns dict
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # Check if it's AutoModel (returns dict with 'logits')
                    outputs = model(input_ids, attention_mask)
                    if isinstance(outputs, dict) or hasattr(outputs, 'logits'):
                        # AutoModelForSequenceClassification returns SequenceClassifierOutput
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs['logits']
                    else:
                        # Regular PyTorch model
                        logits = outputs
                    
                elif isinstance(batch, list) and len(batch) == 2: # TensorDataset or SpeakerDataset
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    logits = model(inputs)
                else:
                    print("Error: Unknown batch format")
                    continue

                _, predicted = torch.max(logits, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        # 4. Calculate Metrics
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        # Infer architecture/embedding names for report
        arch_name = model_name.split(' - ')[0] if ' - ' in model_name else model_name
        emb_name = model_name.split(' - ')[1] if ' - ' in model_name else 'N/A'

        results[model_name] = {
            'f1_score': f1,
            'architecture': arch_name,
            'embedding': emb_name
        }
        print(f"  ✅ F1 Score (Macro): {f1:.4f}")

    except Exception as e:
        print(f"  ❌ Error evaluando {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

print()
print("RESULTADOS FINALES - ANÁLISIS COMPLETO")
print("=" * 75)

if results:
    # Crear DataFrame con resultados
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results = df_results.sort_values('f1_score', ascending=False)

    print(f"\n✅ Se evaluaron {len(df_results)} modelos exitosamente")
    print("\nRanking completo de modelos por F1 Score:")
    print("-" * 75)
    for i, (model_name, row) in enumerate(df_results.iterrows(), 1):
        print(f"{i:2d}. {model_name:35s} F1: {row['f1_score']:.4f} ({row['architecture']} / {row['embedding']})")

    print()

    # Top 5 mejores modelos
    top_5 = df_results.head(5)

    # Bottom 5 peores modelos
    bottom_5 = df_results.tail(5)

    # Gráfico Top 5
    plt.figure(figsize=(12, 8))

    colors_top = ['#2E8B57', '#32CD32', '#00FF7F', '#7CFC00', '#ADFF2F']  # Verdes
    bars_top = plt.bar(range(len(top_5)), top_5['f1_score'], color=colors_top, alpha=0.8)

    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('F1 Score (Macro)', fontsize=12)
    plt.title('TOP 5 - Mejores Modelos (Análisis Completo)\nClasificación de Hablantes - F1 Score', fontsize=14, pad=20)
    plt.ylim([0, 1])

    plt.xticks(range(len(top_5)), top_5.index, rotation=45, ha='right')

    for bar, score in zip(bars_top, top_5['f1_score']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('imagenes/analisis/clasificacion_hablantes/top_5_modelos_deep_learning_f1.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid blocking

    # Gráfico Bottom 5
    plt.figure(figsize=(12, 8))

    colors_bottom = ['#FFA500', '#FF7F50', '#FF6347', '#FF0000', '#DC143C']  # Naranjas/Rojos (de claro a oscuro)
    bars_bottom = plt.bar(range(len(bottom_5)), bottom_5['f1_score'], color=colors_bottom, alpha=0.8)

    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('F1 Score (Macro)', fontsize=12)
    plt.title('BOTTOM 5 - Peores Modelos (Análisis Completo)\nClasificación de Hablantes - F1 Score', fontsize=14, pad=20)
    plt.ylim([0, 1])

    plt.xticks(range(len(bottom_5)), bottom_5.index, rotation=45, ha='right')

    for bar, score in zip(bars_bottom, bottom_5['f1_score']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('imagenes/analisis/clasificacion_hablantes/bottom_5_modelos_deep_learning_f1.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid blocking

    print("\n" + "=" * 75)
    print("✅ GRÁFICOS GENERADOS EXITOSAMENTE")
    print("=" * 75)
    print("Ubicación: imagenes/analisis/clasificacion_hablantes/")
    print("  1. top_5_modelos_deep_learning_f1.png")
    print("  2. bottom_5_modelos_deep_learning_f1.png")
    print("=" * 75)

    # Análisis por arquitectura
    print("\nANÁLISIS POR ARQUITECTURA (F1 Score)")
    print("-" * 75)
    architecture_stats = df_results.groupby('architecture')['f1_score'].agg(['mean', 'max', 'min', 'count'])
    architecture_stats.columns = ['F1 Promedio', 'F1 Máximo', 'F1 Mínimo', 'Cantidad']
    print(architecture_stats.round(4).to_string())

    print("\nANÁLISIS POR TIPO DE EMBEDDING (F1 Score)")
    print("-" * 75)
    embedding_stats = df_results.groupby('embedding')['f1_score'].agg(['mean', 'max', 'min', 'count'])
    embedding_stats.columns = ['F1 Promedio', 'F1 Máximo', 'F1 Mínimo', 'Cantidad']
    print(embedding_stats.round(4).to_string())

    print("\nTOP 5 MEJORES MODELOS:")
    print("-" * 75)
    for i, (name, row) in enumerate(top_5.iterrows(), 1):
        print(f"{i}. {name:35s} F1: {row['f1_score']:.4f}  ({row['architecture']} / {row['embedding']})")

    print("\nBOTTOM 5 PEORES MODELOS:")
    print("-" * 75)
    for i, (name, row) in enumerate(bottom_5.iterrows(), 1):
        print(f"{i}. {name:35s} F1: {row['f1_score']:.4f}  ({row['architecture']} / {row['embedding']})")

else:
    print("❌ No se pudieron evaluar modelos. Verifica que:")
    print("   - Los archivos de modelos existen en models/clasificacion_hablantes/")
    print("   - Los vectorizadores existen en models/")
    print("   - Los datasets están disponibles")
