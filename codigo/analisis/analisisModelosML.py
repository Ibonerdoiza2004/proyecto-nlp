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
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gensim.models import FastText, Word2Vec
import pickle
import joblib

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.style.use('default')

print("ANÁLISIS COMPARATIVO COMPLETO DE MODELOS DE DEEP LEARNING")
print("=" * 70)

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

# Preparar datos para BERT
df_bert = pd.read_csv("dataset/dataset_bert.csv")
df_bert = df_bert[df_bert["text"].str.len() >= 10].copy()
texts_bert = df_bert["text"].tolist()
labels_bert = df_bert["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels_bert)
num_classes = len(label_encoder.classes_)

# Split para BERT
X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
    texts_bert, labels_encoded, test_size=0.2, random_state=10, stratify=labels_encoded
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
print()

# Definir todas las clases de modelos
import torch.nn as nn

# CNN + BERT CLS
class CNNBertCLSClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.ln1(self.fc1(x))))
        x = self.dropout(self.relu(self.ln2(self.fc2(x))))
        return self.fc3(x)

# CNN + BERT (mean pooling)
class CNNEmbeddingsClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128 * (embedding_dim // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, embedding_dim]
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)

# CNN-LSTM + BERT CLS (checkpoint dimensions)
class CNNLSTMBertCLSClassifier_Checkpoint(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, 64)  # 64 instead of 128
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)  # 64 instead of 128
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64 * 2, num_classes)  # 128 instead of 256

    def forward(self, x):
        projected = self.projection(x).unsqueeze(1)
        seq = projected.repeat(1, 5, 1)
        lstm_out, (hidden, cell) = self.lstm(seq)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# CNN-LSTM + BERT (checkpoint dimensions) - Simplified
class CNNLSTMBERTClassifier_Checkpoint(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, 64, kernel_size=k)  # 64 filters, input is embedding_dim
            for k in [2, 3, 4]
        ])
        self.lstm = nn.LSTM(192, 128, batch_first=True, bidirectional=True)  # 192 = 3*64
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 2, num_classes)  # 256

    def forward(self, x):
        # x: [batch, 768] - single embedding
        x = x.unsqueeze(1).repeat(1, 10, 1)  # [batch, 10, 768] - create sequence
        x = x.transpose(1, 2)  # [batch, 768, 10]

        conv_results = []
        for conv in self.convs:
            conv_result = torch.relu(conv(x))  # [batch, 64, 9], [batch, 64, 8], [batch, 64, 7]
            pooled = torch.max(conv_result, dim=2)[0]  # [batch, 64]
            conv_results.append(pooled)
        x = torch.cat(conv_results, dim=1).unsqueeze(1)  # [batch, 1, 192]
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# LSTM + BERT with attention (checkpoint dimensions)
class BERTLSTMClassifier_Checkpoint(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, 256, num_layers=4, batch_first=True,  # 4 layers, 256 hidden
                           dropout=dropout if 4 > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        # Attention mechanism
        self.attention = nn.ModuleDict({
            'Wa': nn.Linear(512, 512),  # 512 = 256*2
            'Ua': nn.Linear(512, 512),
            'Va': nn.Linear(512, 1)
        })

        self.fc = nn.Linear(512, num_classes)  # 512 instead of 256

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

# CNN-LSTM + FastText (checkpoint dimensions - 64 filters, with batch norm)
class CNNLSTMClassifier_Checkpoint(nn.Module):
    def __init__(self, embedding_matrix, num_classes, dropout):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, 64, kernel_size=k)  # 64 filters
            for k in [2, 3, 4]
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(64)  # 64
            for _ in [2, 3, 4]
        ])

        self.lstm = nn.LSTM(192, 128, batch_first=True, bidirectional=True)  # 192 = 3*64
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 2, num_classes)  # 256

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_results = []
        for conv, bn in zip(self.convs, self.batch_norms):
            conv_result = torch.relu(bn(conv(x)))
            pooled = torch.max(conv_result, dim=2)[0]
            conv_results.append(pooled)
        x = torch.cat(conv_results, dim=1).unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# BiGRU + Word2Vec (checkpoint dimensions - 2 layers)
class BiGRUClassifier_Checkpoint(nn.Module):
    def __init__(self, embedding_matrix, num_classes, dropout):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(embedding_dim, 128, num_layers=2, batch_first=True,  # 2 layers, 128 hidden
                         dropout=dropout if 2 > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 2, num_classes)  # 256

    def forward(self, x):
        x = self.embedding(x)
        gru_out, hidden = self.gru(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# MLP Classifier for Perceptrons (BoW, TF-IDF)
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

# CNN-LSTM + FastText
class CNNLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, num_filters, kernel_sizes, hidden_dim, num_classes, dropout):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k)
            for k in kernel_sizes
        ])

        self.lstm = nn.LSTM(num_filters * len(kernel_sizes), hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conv_results = []
        for conv in self.convs:
            conv_result = torch.relu(conv(x))
            pooled = torch.max(conv_result, dim=2)[0]
            conv_results.append(pooled)
        x = torch.cat(conv_results, dim=1).unsqueeze(1)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# BiLSTM + FastText
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

# BiGRU + FastText
class BiGRUClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                         dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        gru_out, hidden = self.gru(x)
        hidden_concat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_concat))

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
    all_words = [word for text in texts for word in text]
    vocab = set(all_words)
    vocab_size = len(vocab) + 2
    word2idx = {word: idx+2 for idx, word in enumerate(vocab)}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1

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
    vocab = set(all_words)
    
    # Crear vocab básico
    word2idx = {word: idx+2 for idx, word in enumerate(list(vocab)[:target_vocab_size-2])}
    word2idx['<pad>'] = 0
    word2idx['<unk>'] = 1
    
    # Rellenar vocab hasta target_vocab_size si es necesario
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
        if word in ['<pad>', '<unk>'] or word.startswith('<extra_'):
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

# Modelos a evaluar
model_configs = {
    # BERT CLS models
    'CNN + BERT CLS': {
        'path': 'models/best_cnn_bert_cls.pth',
        'type': 'bert_cls',
        'architecture': 'CNN',
        'embedding': 'BERT CLS'
    },
    'CNN-LSTM + BERT CLS': {
        'path': 'models/best_cnnlstm_bert_cls.pth',
        'type': 'bert_cls',
        'architecture': 'CNN-LSTM',
        'embedding': 'BERT CLS'
    },
    'LSTM + BERT CLS': {
        'path': 'models/best_lstm_bert_cls.pth',
        'type': 'bert_cls',
        'architecture': 'LSTM',
        'embedding': 'BERT CLS'
    },
    'GRU + BERT CLS': {
        'path': 'models/best_gru_bert_cls.pth',
        'type': 'bert_cls',
        'architecture': 'GRU',
        'embedding': 'BERT CLS'
    },

    # BERT mean pooling models
    'CNN + BERT': {
        'path': 'models/cnn_bert.pth',
        'type': 'bert_mean',
        'architecture': 'CNN',
        'embedding': 'BERT Mean'
    },
    'CNN-LSTM + BERT': {
        'path': 'models/best_cnnlstm_bert.pth',
        'type': 'bert_mean',
        'architecture': 'CNN-LSTM',
        'embedding': 'BERT Mean'
    },
    'LSTM + BERT': {
        'path': 'models/best_bert_speaker.pth',
        'type': 'bert_mean',
        'architecture': 'LSTM',
        'embedding': 'BERT Mean'
    },
    'GRU + BERT': {
        'path': 'models/gru_bert.pth',
        'type': 'bert_mean',
        'architecture': 'GRU',
        'embedding': 'BERT Mean'
    },

    # FastText models
    'CNN + FastText': {
        'path': 'models/cnn_fasttext.pth',
        'type': 'fasttext',
        'architecture': 'CNN',
        'embedding': 'FastText'
    },
    'CNN-LSTM + FastText': {
        'path': 'models/best_cnnlstm_fasttext.pth',
        'type': 'fasttext',
        'architecture': 'CNN-LSTM',
        'embedding': 'FastText'
    },
    'LSTM + FastText': {
        'path': 'models/bilstm_fasttext.pth',
        'type': 'fasttext',
        'architecture': 'BiLSTM',
        'embedding': 'FastText'
    },
    'GRU + FastText': {
        'path': 'models/best_gru_fasttext.pth',
        'type': 'fasttext',
        'architecture': 'BiGRU',
        'embedding': 'FastText'
    },

    # Word2Vec models
    'CNN-LSTM + Word2Vec': {
        'path': 'models/cnn_lstm_w2v.pth',
        'type': 'word2vec',
        'architecture': 'CNN-LSTM',
        'embedding': 'Word2Vec'
    },
    'GRU + Word2Vec': {
        'path': 'models/bigru_w2v.pth',
        'type': 'word2vec',
        'architecture': 'BiGRU',
        'embedding': 'Word2Vec'
    },

    'LSTM Word2Vec': {
        'path': 'models/word2vec_lstm.pth',
        'type': 'word2vec_speaker',
        'architecture': 'BiLSTM',
        'embedding': 'Word2Vec'
    },
    'CNN + Word2Vec': {
        'path': 'models/cnn_speaker_classifier.pth',
        'type': 'word2vec_speaker',
        'architecture': 'CNN',
        'embedding': 'Word2Vec'
    },

    # BERT Speaker Classification
    'LSTM + BERT': {
        'path': 'models/bert_lstm_speaker_classifier.pth',
        'type': 'bert_speaker',
        'architecture': 'BiLSTM+Attention',
        'embedding': 'BERT Mean'
    },

    # Shallow Machine Learning models
    'Perceptron BoW': {
        'path': 'models/best_perceptron_bow.joblib',
        'type': 'perceptron',
        'architecture': 'MLP',
        'embedding': 'BoW'
    },
    'Perceptron TF-IDF': {
        'path': 'models/best_perceptron_tfidf.joblib',
        'type': 'perceptron',
        'architecture': 'MLP',
        'embedding': 'TF-IDF'
    },
    'Perceptron TF-IDF Char': {
        'path': 'models/best_perceptron_tfidf_char.joblib',
        'type': 'perceptron',
        'architecture': 'MLP',
        'embedding': 'TF-IDF Char'
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

        # Evaluar según tipo de modelo
        if config['type'] in ['bert_cls', 'bert_mean']:
            # Modelos BERT
            embedding_type = config['type'].replace('bert_', '')  # 'cls' o 'mean'
            X_test_embeddings = load_bert_embeddings(embedding_type)
            test_dataset = TensorDataset(X_test_embeddings, torch.LongTensor(y_test_bert))
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            # Crear modelo según arquitectura
            if 'cnn_bert_cls' in config['path']:
                model = CNNBertCLSClassifier(768, num_classes, 0.5).to(device)
            elif 'cnnlstm_bert_cls' in config['path']:
                model = CNNLSTMBertCLSClassifier_Checkpoint(768, num_classes, 0.5).to(device)  # Use checkpoint version
            elif 'lstm_bert_cls' in config['path']:
                model = LSTMBertCLSClassifier(768, 128, 2, num_classes, 0.3).to(device)
            elif 'gru_bert_cls' in config['path']:
                model = GRUBertCLSClassifier(768, 128, 2, num_classes, 0.3).to(device)
            elif 'cnn_bert' in config['path']:
                model = CNNEmbeddingsClassifier(768, num_classes, 0.5).to(device)
            elif 'cnnlstm_bert' in config['path']:
                model = CNNLSTMBERTClassifier_Checkpoint(768, num_classes, 0.5).to(device)  # Use checkpoint version
            elif 'bert_speaker' in config['path']:
                model = BERTLSTMClassifier_Checkpoint(768, num_classes, 0.3).to(device)  # Use checkpoint version
            elif 'gru_bert' in config['path']:
                model = GRUEmbeddingsClassifier(768, 128, 2, num_classes, 0.3).to(device)
            else:
                print(f"  ⚠️  Arquitectura BERT no implementada para {model_name}")
                continue

        elif config['type'] in ['word2vec_speaker']:
            # Modelos de speaker classification (Word2Vec embeddings)
            word2idx, embedding_matrix, max_length = create_text_vocab_and_embeddings_fixed('word2vec', 3701, 200)
            test_dataset = TextDataset(X_test_texts, y_test_texts, word2idx, max_length)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # Crear modelo según arquitectura
            if 'word2vec_lstm' in config['path']:
                model = LSTMSpeakerClassifier(embedding_matrix, num_classes, 0.3).to(device)
            elif 'cnn_speaker_classifier' in config['path']:
                model = CNNSpeakerClassifier(embedding_matrix, num_classes, 0.3).to(device)
            else:
                print(f"  ⚠️  Arquitectura speaker no implementada para {model_name}")
                continue

        elif config['type'] in ['bert_speaker']:
            # Modelos BERT speaker
            X_test_embeddings = load_bert_embeddings('mean')
            test_dataset = TensorDataset(X_test_embeddings, torch.LongTensor(y_test_bert))
            test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

            model = BERTLSTMSpeakerClassifier(768, num_classes, 0.3).to(device)

        elif config['type'] in ['perceptron']:
            # Modelos Perceptrón (MLP)
            if 'bow' in config['path']:
                # Perceptron BoW
                vectorizer = joblib.load('models/vec_bow.joblib')
                X_test_transformed = vectorizer.transform(X_test_texts_bow).toarray()
                num_features = X_test_transformed.shape[1]
                model = MLPClassifier(num_features, num_classes).to(device)
                test_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_test_transformed), 
                    torch.LongTensor(y_test_texts)
                )
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            elif 'tfidf' in config['path'] and 'char' not in config['path']:
                # Perceptron TF-IDF
                vectorizer = joblib.load('models/vec_tfidf_word.joblib')
                X_test_transformed = vectorizer.transform(X_test_texts_bow).toarray()
                num_features = X_test_transformed.shape[1]
                model = MLPClassifier(num_features, num_classes).to(device)
                test_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_test_transformed), 
                    torch.LongTensor(y_test_texts)
                )
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            elif 'tfidf_char' in config['path']:
                # Perceptron TF-IDF Char
                vectorizer = joblib.load('models/vec_tfidf_char.joblib')
                X_test_transformed = vectorizer.transform(X_test_texts_bow).toarray()
                num_features = X_test_transformed.shape[1]
                model = MLPClassifier(num_features, num_classes).to(device)
                test_dataset = torch.utils.data.TensorDataset(
                    torch.FloatTensor(X_test_transformed), 
                    torch.LongTensor(y_test_texts)
                )
                test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            else:
                print(f"  ⚠️  Tipo de perceptrón no implementado para {model_name}")
                continue

        elif config['type'] in ['fasttext', 'word2vec']:
            # Modelos FastText/Word2Vec
            model_type = config['type']
            
            # Casos especiales con vocabs fijos
            if ('cnnlstm' in config['path'].lower() or 'cnn_lstm' in config['path'].lower()) and 'fasttext' in config['path']:
                # CNN-LSTM FastText: vocab_size=16442
                vocab_size = 16442
                embedding_dim = 200
                word2idx, embedding_matrix, max_length = create_text_vocab_and_embeddings_fixed(model_type, vocab_size, embedding_dim)
            elif ('cnnlstm' in config['path'].lower() or 'cnn_lstm' in config['path'].lower()) and 'w2v' in config['path']:
                # CNN-LSTM Word2Vec: vocab_size=3701 (mismo que BiGRU Word2Vec)
                vocab_size = 3701
                embedding_dim = 200
                word2idx, embedding_matrix, max_length = create_text_vocab_and_embeddings_fixed(model_type, vocab_size, embedding_dim)
            elif 'bigru' in config['path'].lower() and 'w2v' in config['path']:
                # BiGRU Word2Vec: vocab_size=3701
                vocab_size = 3701
                embedding_dim = 200
                word2idx, embedding_matrix, max_length = create_text_vocab_and_embeddings_fixed(model_type, vocab_size, embedding_dim)
            else:
                # Vocab dinámico normal para otros modelos
                word2idx, embedding_matrix, max_length = create_text_vocab_and_embeddings(model_type)

            test_dataset = TextDataset(X_test_texts, y_test_texts, word2idx, max_length)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            vocab_size, embedding_dim = embedding_matrix.shape

            # Crear modelo según arquitectura
            if 'cnn' in config['path'].lower() and 'lstm' not in config['path'].lower():
                # CNN only (FastText/Word2Vec)
                model = CNNClassifier_Checkpoint(embedding_matrix, num_classes, 0.5).to(device)
            elif 'cnnlstm' in config['path'].lower() or 'cnn_lstm' in config['path'].lower():
                # CNN-LSTM (FastText/Word2Vec)
                model = CNNLSTMClassifier_Checkpoint(embedding_matrix, num_classes, 0.5).to(device)
            elif 'bilstm' in config['path'].lower():
                model = BiLSTMClassifier(embedding_matrix, 128, 2, num_classes, 0.3).to(device)
            elif 'bigru' in config['path'].lower() or 'gru' in config['path'].lower():
                # BiGRU (FastText/Word2Vec)
                if 'w2v' in config['path'] or 'word2vec' in config['path']:
                    model = BiGRUClassifier_Checkpoint(embedding_matrix, num_classes, 0.3).to(device)
                else:
                    model = BiGRUClassifier(embedding_matrix, 128, 2, num_classes, 0.3).to(device)
            else:
                print(f"  ⚠️  Arquitectura {config['type']} no implementada para {model_name}")
                continue

        # Cargar pesos del modelo
        if not load_model_weights(model, config['path']):
            continue

        model.eval()

        # Evaluar
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                if config['type'] in ['bert_cls', 'bert_mean']:
                    emb, lbl = batch
                    emb, lbl = emb.to(device), lbl.to(device)
                    outputs = model(emb)
                else:
                    indices, lbl = batch
                    indices, lbl = indices.to(device), lbl.to(device)
                    outputs = model(indices)

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())

        # Calcular accuracy
        if config['type'] in ['bert_cls', 'bert_mean', 'bert_speaker']:
            accuracy = accuracy_score(y_test_bert, all_preds)
        else:
            accuracy = accuracy_score(y_test_texts, all_preds)

        results[model_name] = {
            'accuracy': accuracy,
            'architecture': config['architecture'],
            'embedding': config['embedding']
        }
        print(f"  ✅ Accuracy: {accuracy:.4f}")

    except Exception as e:
        print(f"  ❌ Error evaluando {model_name}: {str(e)}")
        continue

print()
print("RESULTADOS FINALES")
print("=" * 70)

if results:
    # Crear DataFrame con resultados
    df_results = pd.DataFrame.from_dict(results, orient='index')
    df_results = df_results.sort_values('accuracy', ascending=False)

    print("Ranking completo de modelos por accuracy:")
    for i, (model_name, row) in enumerate(df_results.iterrows(), 1):
        print("2d")

    print()

    # Top 5 mejores modelos
    top_5 = df_results.head(5)

    # Bottom 5 peores modelos
    bottom_5 = df_results.tail(5)

    # Gráfico Top 5
    plt.figure(figsize=(12, 8))

    colors_top = ['#2E8B57', '#32CD32', '#00FF7F', '#7CFC00', '#ADFF2F']  # Verdes
    bars_top = plt.bar(range(len(top_5)), top_5['accuracy'], color=colors_top, alpha=0.8)

    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('TOP 5 - Mejores Modelos de Deep Learning\nClasificación de Hablantes', fontsize=14, pad=20)
    plt.ylim([0, 1])

    plt.xticks(range(len(top_5)), top_5.index, rotation=45, ha='right')

    for bar, acc in zip(bars_top, top_5['accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('top_5_modelos_deep_learning.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid blocking

    # Gráfico Bottom 5
    plt.figure(figsize=(12, 8))

    colors_bottom = ['#FFA500', '#FF7F50', '#FF6347', '#FF0000', '#DC143C']  # Naranjas/Rojos (de claro a oscuro)
    bars_bottom = plt.bar(range(len(bottom_5)), bottom_5['accuracy'], color=colors_bottom, alpha=0.8)

    plt.xlabel('Modelos', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('BOTTOM 5 - Peores Modelos de Deep Learning\nClasificación de Hablantes', fontsize=14, pad=20)
    plt.ylim([0, 1])

    plt.xticks(range(len(bottom_5)), bottom_5.index, rotation=45, ha='right')

    for bar, acc in zip(bars_bottom, bottom_5['accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('bottom_5_modelos_deep_learning.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Commented out to avoid blocking

    print("✅ Gráficos guardados:")
    print("   - top_5_modelos_deep_learning.png")
    print("   - bottom_5_modelos_deep_learning.png")

    # Análisis por arquitectura
    print("\nANÁLISIS POR ARQUITECTURA")
    print("-" * 50)
    architecture_stats = df_results.groupby('architecture')['accuracy'].agg(['mean', 'max', 'count'])
    print(architecture_stats.round(4))

    print("\nANÁLISIS POR TIPO DE EMBEDDING")
    print("-" * 50)
    embedding_stats = df_results.groupby('embedding')['accuracy'].agg(['mean', 'max', 'count'])
    print(embedding_stats.round(4))

    print("\nTOP 5 MEJORES MODELOS:")
    print("-" * 50)
    for i, (name, row) in enumerate(top_5.iterrows(), 1):
        print("2d")

    print("\nBOTTOM 5 PEORES MODELOS:")
    print("-" * 50)
    for i, (name, row) in enumerate(bottom_5.iterrows(), 1):
        print("2d")

else:
    print("❌ No se pudieron evaluar modelos")
