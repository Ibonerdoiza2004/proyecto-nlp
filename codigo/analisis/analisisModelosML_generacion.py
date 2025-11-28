"""
Análisis de modelos generado con IAg
"""
import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import Word2Vec, FastText
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definición de Modelos
model_configs = [
    {
        'name': 'GRU + Word2Vec',
        'model_path': 'models/gru_word2vec_text_generator.pth',
        'vocab_path': 'models/vocab_generator.pkl',
        'embedding_type': 'word2vec',
        'model_type': 'gru'
    },
    {
        'name': 'GRU + FastText',
        'model_path': 'models/gru_text_generator.pth',
        'vocab_path': 'models/vocab_generator.pkl',
        'embedding_type': 'fasttext',
        'model_type': 'gru'
    },
    # {
    #     'name': 'LSTM + Word2Vec',
    #     'model_path': 'models/word2vec_lstm.pth',
    #     'vocab_path': 'models/vocab_generator.pkl',
    #     'embedding_type': 'word2vec',
    #     'model_type': 'lstm'
    # },
    {
        'name': 'LSTM + FastText',
        'model_path': 'models/lstm_text_generator.pth',
        'vocab_path': 'models/vocab_generator.pkl',
        'embedding_type': 'fasttext',
        'model_type': 'lstm'
    }
]

# Carga del Modelo de Clasificación de Speaker
print("Cargando modelo de clasificación de speaker...")
perceptron_model = load('models/best_perceptron_tfidf.joblib')
vectorizer = load('models/vec_tfidf_word.joblib')
label_encoder = load('models/label_encoder_speaker.joblib')

# Función para predecir speaker
def predict_speaker(text):
    return "Predicción no disponible (error en modelo)"

# Clase Modelo Base (adaptada de los scripts)
class TextGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, 
                 dropout_p=0.3, pretrained_embeddings=None, padding_idx=0, model_type='gru'):
        super(TextGenerator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.padding_idx = padding_idx
        self.model_type = model_type
        
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, 
                                         padding_idx=padding_idx, _weight=pretrained_embeddings)
        
        if model_type == 'gru':
            self.gru = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_p if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        elif model_type == 'lstm':
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout_p if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
        
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x_in, hidden=None):
        embedded = self.embedding(x_in)
        if self.model_type == 'gru':
            rnn_out, hidden = self.gru(embedded, hidden)
        elif self.model_type == 'lstm':
            rnn_out, hidden = self.lstm(embedded, hidden)
        last_output = rnn_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        
        return output, hidden
    
    def generate_text(self, start_text, vocab, idx_to_word, seq_length, max_length=50, temperature=1.0):
        self.eval()
        context = [vocab.get(word, vocab["<UNK>"]) for word in start_text]
        generated = start_text.copy()
        
        with torch.no_grad():
            hidden = self.init_hidden(1)
            
            for _ in range(max_length):
                if len(context) > seq_length:
                    input_seq = context[-seq_length:]
                else:
                    input_seq = [vocab["<PAD>"]] * (seq_length - len(context)) + context
                
                input_tensor = torch.LongTensor([input_seq]).to(device)
                output, hidden = self(input_tensor, hidden)
                
                output = output / temperature
                probs = torch.softmax(output, dim=1)
                next_idx = torch.multinomial(probs, 1).item()
                
                if next_idx == vocab["<END>"]:
                    break
                
                next_word = idx_to_word.get(next_idx, "<UNK>")
                if next_word not in ["<PAD>", "<UNK>", "<START>"]:
                    generated.append(next_word)
                    context.append(next_idx)
        
        return generated

# Función para generar texto
def generate_text(model, start_text, vocab, idx_to_word, seq_length, max_length=50, temperature=1.0):
    model.eval()
    context = [vocab.get(word, vocab["<UNK>"]) for word in start_text]
    generated = start_text.copy()
    
    with torch.no_grad():
        if model.model_type == 'gru':
            hidden = torch.zeros(model.num_layers, 1, model.hidden_dim).to(device)
        elif model.model_type == 'lstm':
            h0 = torch.zeros(model.num_layers, 1, model.hidden_dim).to(device)
            c0 = torch.zeros(model.num_layers, 1, model.hidden_dim).to(device)
            hidden = (h0, c0)
        
        for _ in range(max_length):
            if len(context) > seq_length:
                input_seq = context[-seq_length:]
            else:
                input_seq = [vocab["<PAD>"]] * (seq_length - len(context)) + context
            
            input_tensor = torch.LongTensor([input_seq]).to(device)
            output, hidden = model(input_tensor, hidden)
            
            output = output / temperature
            probs = torch.softmax(output, dim=1)
            next_idx = torch.multinomial(probs, 1).item()
            
            if next_idx == vocab["<END>"]:
                break
            
            next_word = idx_to_word.get(next_idx, "<UNK>")
            if next_word not in ["<PAD>", "<UNK>", "<START>"]:
                generated.append(next_word)
                context.append(next_idx)
    
    return generated

# Función para cargar modelo
def load_model(config):
    with open(config['vocab_path'], 'rb') as f:
        vocab_data = pickle.load(f)
    vocab = vocab_data['vocab']
    idx_to_word = vocab_data['idx_to_word']
    seq_length = vocab_data['seq_length']
    
    checkpoint = torch.load(config['model_path'], map_location=device)
    vocab_size = checkpoint['vocab_size']
    embedding_dim = checkpoint['embedding_dim']
    hidden_dim = checkpoint['hidden_dim']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    
    model = TextGenerator(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_p=dropout,
        model_type=config['model_type']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, idx_to_word, seq_length

# Inicios de frase
start_texts = [
    ["athletic", "ganar"],
    ["balón", "oro"],
    ["trampa"]
]

temperatures = [0.2, 0.7, 1.0]

# Cargar modelos
loaded_models = {}
for config in model_configs:
    try:
        model, vocab, idx_to_word, seq_length = load_model(config)
        loaded_models[config['name']] = {
            'model': model,
            'vocab': vocab,
            'idx_to_word': idx_to_word,
            'seq_length': seq_length
        }
        print(f"Modelo {config['name']} cargado exitosamente.")
    except Exception as e:
        print(f"Error cargando {config['name']}: {e}")

# Generación y Análisis
results = {}
lengths = {name: [] for name in loaded_models.keys()}

for start in start_texts:
    print(f"\n{'='*60}")
    print(f"INICIO: {' '.join(start)}")
    print(f"{'='*60}")
    
    for name, model_data in loaded_models.items():
        print(f"\n--- {name} ---")
        for temp in temperatures:
            generated = generate_text(
                model_data['model'], start, model_data['vocab'], 
                model_data['idx_to_word'], model_data['seq_length'], 
                max_length=20, temperature=temp
            )
            generated_text = ' '.join(generated)
            lengths[name].append(len(generated))
            
            # Predecir speaker
            predicted_speaker = predict_speaker(generated_text)
            
            print(f"  T={temp}: {generated_text}")
            print(f"    Speaker predicho: {predicted_speaker}")
            
            if name not in results:
                results[name] = {}
            if tuple(start) not in results[name]:
                results[name][tuple(start)] = {}
            results[name][tuple(start)][temp] = {
                'text': generated_text,
                'speaker': predicted_speaker
            }

# Gráficos
plt.figure(figsize=(12, 6))

# Gráfico de longitudes promedio
avg_lengths = {name: np.mean(lengths[name]) for name in lengths}
plt.subplot(1, 2, 1)
plt.bar(avg_lengths.keys(), avg_lengths.values(), color='skyblue')
plt.title('Longitud Promedio de Texto Generado')
plt.ylabel('Longitud (palabras)')
plt.xticks(rotation=45)

# Gráfico de distribución de speakers predichos
speakers_count = {}
for name, starts in results.items():
    for start, temps in starts.items():
        for temp, data in temps.items():
            speaker = data['speaker']
            key = f"{name} ({temp})"
            if key not in speakers_count:
                speakers_count[key] = {}
            if speaker not in speakers_count[key]:
                speakers_count[key][speaker] = 0
            speakers_count[key][speaker] += 1

# Simplificar: contar total por modelo
total_speakers = {}
for name in loaded_models.keys():
    total_speakers[name] = {}
    for start in start_texts:
        for temp in temperatures:
            if name in results and tuple(start) in results[name] and temp in results[name][tuple(start)]:
                speaker = results[name][tuple(start)][temp]['speaker']
                if speaker not in total_speakers[name]:
                    total_speakers[name][speaker] = 0
                total_speakers[name][speaker] += 1

plt.subplot(1, 2, 2)
speakers_df = pd.DataFrame(total_speakers).fillna(0)
speakers_df.plot(kind='bar', ax=plt.gca())
plt.title('Distribución de Speakers Predichos por Modelo')
plt.ylabel('Conteo')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('graficos/analisis_generacion_texto.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAnálisis completado. Gráfico guardado en 'graficos/analisis_generacion_texto.png'")
