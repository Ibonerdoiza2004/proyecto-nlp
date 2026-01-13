import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import pickle
import argparse


# CONFIGURACIÓN
class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    DATASET_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'dataset_unificado.csv')
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', 'generacion_texto')
    MODEL_PATH = os.path.join(MODEL_DIR, 'enc_dec_gru_bert.pt')
    VOCAB_PATH = os.path.join(MODEL_DIR, 'enc_dec_gru_bert_vocab.pkl')
    
    # BETO
    BETO_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
    ENCODER_DIM = 768  # Dimensión de salida de BETO
    
    # GRU
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    
    # Entrenamiento
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.001
    MAX_INPUT_LEN = 64
    MAX_OUTPUT_LEN = 100
    TEACHER_FORCING_RATIO = 0.5
    
    # Early stopping
    PATIENCE = 5
    
    # Fine-tuning completo
    FINETUNE_EPOCHS = 20
    FINETUNE_LR = 1e-5
    
    # Generación
    TEMPERATURE = 0.8
    TOP_K = 40
    
    # Dispositivo
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# VOCABULARIO
class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        self.word2idx = {token: idx for idx, token in enumerate(special_tokens)}
        
        idx = len(special_tokens)
        for word, count in word_counts.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        print(f"Vocabulario: {self.vocab_size} palabras")
        
    def encode(self, text, max_len=None):
        words = text.lower().split()
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        if max_len:
            indices = indices[:max_len-1]
        indices.append(self.word2idx['<EOS>'])
        return indices
    
    def decode(self, indices):
        words = []
        for idx in indices:
            if idx == self.word2idx['<EOS>']:
                break
            if idx not in [self.word2idx['<PAD>'], self.word2idx['<SOS>']]:
                words.append(self.idx2word.get(idx, '<UNK>'))
        return ' '.join(words)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))


# DATASET
class EncoderDecoderDataset(Dataset):
    def __init__(self, texts, vocab, tokenizer, max_input_len, max_output_len):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len
        self.pairs = self._create_pairs(texts)
        
    def _create_pairs(self, texts):
        pairs = []
        for i in range(len(texts) - 1):
            input_text = texts[i]
            output_text = texts[i + 1]
            
            if len(input_text.split()) >= 5 and len(output_text.split()) >= 5:
                pairs.append((input_text, output_text))
        
        for text in texts:
            if len(text.split()) >= 10:
                words = text.split()
                mid = len(words) // 2
                pairs.append((' '.join(words[:mid]), ' '.join(words[mid:])))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        input_text, output_text = self.pairs[idx]
        
        encoded = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        output_indices = self.vocab.encode(output_text, self.max_output_len)
        
        if len(output_indices) < self.max_output_len:
            output_indices += [self.vocab.word2idx['<PAD>']] * (self.max_output_len - len(output_indices))
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'output': torch.tensor(output_indices, dtype=torch.long)
        }


# ENCODER
class BETOEncoder(nn.Module):
    def __init__(self, hidden_dim, freeze_bert=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config.BETO_MODEL, add_pooling_layer=False)
        
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            for param in self.bert.encoder.layer[-2:].parameters():
                param.requires_grad = True
        
        self.projection = nn.Linear(Config.ENCODER_DIM, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        projected = self.projection(self.dropout(cls_output))
        return projected
    
    def unfreeze_all(self):
        for param in self.bert.parameters():
            param.requires_grad = True
        print(" BERT completamente descongelado para fine-tuning")


# DECODER
class GRUDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, input_token, hidden, encoder_output):
        
        embedded = self.dropout(self.embedding(input_token))
        
        encoder_output = encoder_output.unsqueeze(1)
        gru_input = torch.cat([embedded, encoder_output], dim=2)
        
        output, hidden = self.gru(gru_input, hidden)
        logits = self.fc(output.squeeze(1))
        
        return logits, hidden
    
    def init_hidden(self, batch_size, device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)


# MODELO ENCODER-DECODER
class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.encoder = BETOEncoder(hidden_dim)
        self.decoder = GRUDecoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.hidden_dim = hidden_dim
        
    def forward(self, input_ids, attention_mask, target, teacher_forcing_ratio=0.5):
        batch_size = input_ids.size(0)
        max_len = target.size(1)
        vocab_size = self.decoder.vocab_size
        
        # Encoder
        encoder_output = self.encoder(input_ids, attention_mask)
        
        hidden = encoder_output.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        # Decoder
        outputs = torch.zeros(batch_size, max_len, vocab_size).to(input_ids.device)
        decoder_input = torch.full((batch_size, 1), 2, dtype=torch.long).to(input_ids.device)  # <SOS> = 2
        
        for t in range(max_len):
            output, hidden = self.decoder(decoder_input, hidden, encoder_output)
            outputs[:, t, :] = output
            
            # Teacher forcing
            if np.random.random() < teacher_forcing_ratio:
                decoder_input = target[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(1).unsqueeze(1)
        
        return outputs


# ENTRENAMIENTO
def load_miguel_data():
    df = pd.read_csv(Config.DATASET_PATH)
    miguel_df = df[df['speaker'] == 'MIGUEL']
    texts = miguel_df['text'].tolist()
    return texts


def train_epoch(model, dataloader, criterion, optimizer, device, teacher_forcing_ratio):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target = batch['output'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, target, teacher_forcing_ratio)
        
        # Calcular loss
        outputs = outputs.view(-1, outputs.size(-1))
        target = target.view(-1)
        loss = criterion(outputs, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['output'].to(device)
            
            outputs = model(input_ids, attention_mask, target, teacher_forcing_ratio=0)
            
            outputs = outputs.view(-1, outputs.size(-1))
            target = target.view(-1)
            loss = criterion(outputs, target)
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_model():
    print("ENCODER-DECODER: BETO (Encoder) + GRU (Decoder)")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    
    # Cargar datos
    texts = load_miguel_data()
    
    # Crear vocabulario
    vocab = Vocabulary(min_freq=2)
    vocab.build_vocab(texts)
    vocab.save(Config.VOCAB_PATH)
    
    # Tokenizer de BETO
    tokenizer = AutoTokenizer.from_pretrained(Config.BETO_MODEL)
    
    # Split datos
    split_idx = int(len(texts) * 0.9)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    # Datasets
    train_dataset = EncoderDecoderDataset(train_texts, vocab, tokenizer, 
                                          Config.MAX_INPUT_LEN, Config.MAX_OUTPUT_LEN)
    val_dataset = EncoderDecoderDataset(val_texts, vocab, tokenizer,
                                        Config.MAX_INPUT_LEN, Config.MAX_OUTPUT_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # Modelo
    model = EncoderDecoder(
        vocab_size=vocab.vocab_size,
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    # Criterio y optimizador
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    
    # FASE 1: BERT CONGELADO
    print("FASE 1: Entrenamiento con BERT congelado")

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        print(f"\n[FASE 1] Epoch {epoch + 1}/{Config.EPOCHS}")
        
        # Reducir teacher forcing gradualmente
        tf_ratio = max(0.2, Config.TEACHER_FORCING_RATIO - epoch * 0.02)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE, tf_ratio)
        val_loss = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        scheduler.step(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Teacher Forcing: {tf_ratio:.2f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab.vocab_size,
                'embedding_dim': Config.EMBEDDING_DIM,
                'hidden_dim': Config.HIDDEN_DIM,
                'num_layers': Config.NUM_LAYERS,
                'dropout': Config.DROPOUT,
                'phase': 1
            }, Config.MODEL_PATH)
            print(f"  Modelo guardado (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping en epoch {epoch + 1}")
                break
        
        # Muestra de generación
        if (epoch + 1) % 3 == 0:
            sample = generate_text(model, vocab, tokenizer, "El partido de hoy")
            print(f"  Muestra: {sample}")
    
    # FASE 2: FINE-TUNING COMPLETO
    print("FASE 2: BERT descongelado")
    
    # Descongelar todo BERT
    model.encoder.unfreeze_all()
    
    # Nuevo optimizador con learning rate más bajo
    optimizer_ft = torch.optim.AdamW(model.parameters(), lr=Config.FINETUNE_LR, weight_decay=0.01)
    scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=Config.FINETUNE_EPOCHS)
    
    best_val_loss_ft = best_val_loss
    patience_counter = 0
    
    for epoch in range(Config.FINETUNE_EPOCHS):
        print(f"\n[FASE 2] Epoch {epoch + 1}/{Config.FINETUNE_EPOCHS}")
        
        tf_ratio = 0.2  # Teacher forcing bajo para fine-tuning
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer_ft, Config.DEVICE, tf_ratio)
        val_loss = evaluate(model, val_loader, criterion, Config.DEVICE)
        
        scheduler_ft.step()
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {scheduler_ft.get_last_lr()[0]:.2e}")
        
        if val_loss < best_val_loss_ft:
            best_val_loss_ft = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab.vocab_size,
                'embedding_dim': Config.EMBEDDING_DIM,
                'hidden_dim': Config.HIDDEN_DIM,
                'num_layers': Config.NUM_LAYERS,
                'dropout': Config.DROPOUT,
                'phase': 2
            }, Config.MODEL_PATH)
            print(f"  Modelo guardado (val_loss: {best_val_loss_ft:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping en epoch {epoch + 1}")
                break
        
        # Muestra de generación
        if (epoch + 1) % 2 == 0:
            sample = generate_text(model, vocab, tokenizer, "El partido de hoy")
            print(f"  Muestra: {sample}")
    
    print("ENTRENAMIENTO COMPLETADO (Fase 1 + Fase 2)")
    print(f"Mejor Val Loss Fase 1: {best_val_loss:.4f}")
    print(f"Mejor Val Loss Fase 2: {best_val_loss_ft:.4f}")


# GENERACIÓN
def generate_text(model, vocab, tokenizer, input_text, max_length=50, temperature=0.8):
    model.eval()
    
    # Tokenizar input
    encoded = tokenizer(
        input_text,
        max_length=Config.MAX_INPUT_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(Config.DEVICE)
    attention_mask = encoded['attention_mask'].to(Config.DEVICE)
    
    with torch.no_grad():
        # Encoder
        encoder_output = model.encoder(input_ids, attention_mask)
        hidden = encoder_output.unsqueeze(0).repeat(model.decoder.num_layers, 1, 1)
        
        # Decoder
        generated = []
        decoder_input = torch.tensor([[vocab.word2idx['<SOS>']]]).to(Config.DEVICE)
        
        for _ in range(max_length):
            output, hidden = model.decoder(decoder_input, hidden, encoder_output)
            
            # Evitar generar tokens especiales no deseados
            output[0, vocab.word2idx['<UNK>']] = -1e9
            output[0, vocab.word2idx['<PAD>']] = -1e9
            
            # Aplicar temperatura y top-k sampling
            logits = output / temperature
            top_k_logits, top_k_indices = torch.topk(logits, Config.TOP_K)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices[0, sampled_idx[0]].item()
            
            if next_token == vocab.word2idx['<EOS>']:
                break
            
            generated.append(next_token)
            decoder_input = torch.tensor([[next_token]]).to(Config.DEVICE)
        
        return vocab.decode(generated)


def load_model():
    vocab = Vocabulary()
    vocab.load(Config.VOCAB_PATH)
    
    tokenizer = AutoTokenizer.from_pretrained(Config.BETO_MODEL)
    
    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
    
    model = EncoderDecoder(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout']
    ).to(Config.DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    return model, vocab, tokenizer


# DEMO
def demo():
    print("DEMO (BETO + GRU)")
    
    try:
        model, vocab, tokenizer = load_model()
        print(" Modelo cargado")
    except FileNotFoundError:
        print(" Modelo no encontrado. Ejecuta: python gru_bert.py --train")
        return
    
    print("\nEscribe una frase para extraer la temática.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        try:
            user_input = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if user_input.lower() == 'salir':
            break
        
        if not user_input:
            continue
        
        response = generate_text(model, vocab, tokenizer, user_input)
        print(f"Generado: {response}\n")
    


# ==================== MAIN ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encoder-Decoder BETO + GRU')
    parser.add_argument('--train', action='store_true', help='Entrenar modelo')
    parser.add_argument('--demo', action='store_true', help='Demo interactiva')
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.demo:
        demo()
    else:
        print("Usa --train para entrenar o --demo para probar")