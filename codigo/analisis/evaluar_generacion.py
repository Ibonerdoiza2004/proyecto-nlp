import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import importlib.util
import pickle
import warnings
import logging
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging as hf_logging

try:
    from peft import PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


# Configuración visual
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Rutas base
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'dataset_unificado.csv')
MODELS_ENC_DEC_DIR = os.path.join(BASE_DIR, 'codigo', 'modelos ML', 'generacion texto', 'encoder_decoder')
MODELS_DEC_ONLY_DIR = os.path.join(BASE_DIR, 'codigo', 'modelos ML', 'generacion texto', 'decoder_only')
IMG_DIR = os.path.join(BASE_DIR, 'graficos', 'comparativa_generacion')
os.makedirs(IMG_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

def load_module(path):
    """Carga dinámica de módulos Python desde ruta"""
    name = os.path.basename(path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def get_validation_data():
    """Obtiene el último 10% de los datos de Miguel (Standard Split)"""
    df = pd.read_csv(DATASET_PATH)
    texts = df[df['speaker'] == 'MIGUEL']['text'].tolist()
    split_idx = int(len(texts) * 0.9)
    return texts[split_idx:]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==================================================================================
# EVALUADORES
# ==================================================================================

def eval_enc_dec_bert(mod_path, texts, model_name):
    """Evaluador para modelos basados en BERT (lstm_bert, transformer_bert, etc)"""
    print(f"Evaluando {model_name}...")
    try:
        mod = load_module(mod_path)
        
        # Cargar Vocabulario
        vocab = mod.Vocabulary()
        if hasattr(vocab, 'load'):
            vocab.load(mod.Config.VOCAB_PATH)
        elif hasattr(vocab, 'load_vocab'): # decoder_only transformer
            vocab.load_vocab(mod.Config.VOCAB_PATH)
        else: # pickle directo
             with open(mod.Config.VOCAB_PATH, 'rb') as f:
                 vocab.__dict__.update(pickle.load(f))
        
        # Vocab Size
        # Normalizar atributo
        if hasattr(vocab, 'vocab_size'): v_size = vocab.vocab_size
        elif hasattr(vocab, 'size'): v_size = vocab.size
        elif hasattr(vocab, 'token2idx'): v_size = len(vocab.token2idx)
        else: v_size = len(vocab.word2idx) if hasattr(vocab, 'word2idx') else 0
        
        # Cargar Modelo
        checkpoint = torch.load(mod.Config.MODEL_PATH, map_location=DEVICE)
        
        # Instanciar arquitectura
        if 'transf' in model_name.lower():
            model = mod.TransformerModel(v_size).to(DEVICE)
            state_dict = checkpoint # Transformer suele guardar directo el state_dict
        else:
            model = mod.EncoderDecoder(v_size, mod.Config.EMBEDDING_DIM, 
                                     mod.Config.HIDDEN_DIM, mod.Config.NUM_LAYERS, mod.Config.DROPOUT).to(DEVICE)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            else:
                state_dict = checkpoint

        # Fix mismatch keys BERT
        if 'bert' in model_name.lower():
             # A veces se guardan keys con prefijo 'bert.' o 'encoder.bert.'
             # Loading flexible
             model.load_state_dict(state_dict, strict=False)
        else:
             model.load_state_dict(state_dict)

        model.eval()
        
        # DataLoader
        tokenizer = AutoTokenizer.from_pretrained(mod.Config.BETO_MODEL)
        
        # Adaptador para el Dataset del módulo
        if 'transf' in model_name.lower(): 
            dataset = mod.Dataset(texts, vocab, tokenizer) # Transformer usa Dataset diferente
            collate_fn = None
        else:
            dataset = mod.EncoderDecoderDataset(texts, vocab, tokenizer, mod.Config.MAX_INPUT_LEN, mod.Config.MAX_OUTPUT_LEN)
            collate_fn = None

        dl = DataLoader(dataset, batch_size=16, shuffle=False)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        total_loss = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch in dl:
                # Estandarizar entradas según el modelo
                if 'transf' in model_name.lower():
                    src, mask, tgt = batch 
                    src, mask, tgt = src.to(DEVICE), mask.to(DEVICE), tgt.to(DEVICE)
                    out = model(src, mask, tgt[:, :-1])
                    loss = criterion(out.reshape(-1, v_size), tgt[:, 1:].reshape(-1))
                else:
                    # LSTM/GRU dataset devuelve dict
                    input_ids = batch['input_ids'].to(DEVICE)
                    mask = batch['attention_mask'].to(DEVICE)
                    target = batch['output'].to(DEVICE)
                    out = model(input_ids, mask, target, 0) # 0 teacher forcing
                    loss = criterion(out.view(-1, v_size), target.view(-1))
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dl)
        ppl = np.exp(avg_loss)
        
        return {
            "Model": model_name,
            "Type": "Encoder-Decoder",
            "PPL": ppl,
            "Params (M)": count_parameters(model) / 1e6
        }

    except Exception as e:
        print(f"Error evaluando {model_name}: {e}")
        return None

def eval_enc_dec_scratch(mod_path, texts, model_name):
    """Evaluador para Transformer Scratch y FastText"""
    print(f"Evaluando {model_name}...")
    try:
        mod = load_module(mod_path)
        
        # Inject Vocabulary to __main__ for Pickle
        sys.modules['__main__'].Vocabulary = mod.Vocabulary
        
        # Vocabulario
        with open(mod.Config.VOCAB_PATH, 'rb') as f:
            if 'transf' in model_name.lower():
                vocab = pickle.load(f) # Transformer scratch guarda el objeto entero
                vocab_size = vocab.size if hasattr(vocab, 'size') else vocab.vocab_size
            else:
                vocab_data = pickle.load(f) # FastText guarda dict
                vocab = mod.Vocabulary()
                vocab.__dict__.update(vocab_data)
                vocab_size = vocab.vocab_size
        
        # Modelo
        checkpoint = torch.load(mod.Config.MODEL_PATH, map_location=DEVICE)
        
        if 'transf' in model_name.lower():
            model = mod.Transformer(vocab_size).to(DEVICE)
            model.load_state_dict(checkpoint)
            # Dataset Transformer Scratch
            # IMPORTANTE: Recrear la logica de TextDataset localmente para asegurar consistencia
            # El TextDataset de transformer_scratch espera (list_texts, vocab)
            # Y hace padding a Config.MAX_LEN
            class ScratchDataset(torch.utils.data.Dataset):
                def __init__(self, texts, vocab):
                    self.pairs = []
                    for i in range(len(texts)-1):
                        if len(texts[i].split()) > 5:
                            self.pairs.append((texts[i], texts[i+1]))
                    self.vocab = vocab
                def __len__(self): return len(self.pairs)
                def __getitem__(self, idx):
                    src, tgt = self.pairs[idx]
                    src_idx = self.vocab.encode(src)[:mod.Config.MAX_LEN]
                    tgt_idx = self.vocab.encode(tgt)[:mod.Config.MAX_LEN]
                    src_idx += [0]*(mod.Config.MAX_LEN - len(src_idx))
                    tgt_idx += [0]*(mod.Config.MAX_LEN - len(tgt_idx))
                    return torch.tensor(src_idx), torch.tensor(tgt_idx)
            
            dataset = ScratchDataset(texts, vocab)
        else:
            model = mod.Seq2Seq(vocab_size, mod.Config.EMBEDDING_DIM, mod.Config.HIDDEN_DIM, 
                              mod.Config.NUM_LAYERS, mod.Config.DROPOUT).to(DEVICE)
            if isinstance(checkpoint, dict):
                 state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
            else:
                 state_dict = checkpoint
            model.load_state_dict(state_dict)
                 
            # Dataset FastText (usa lógica de listas simple)
            class SimpleDataset(torch.utils.data.Dataset):
                def __init__(self, texts, vocab, max_len):
                    self.pairs = []
                    for i in range(len(texts)-1):
                        if len(texts[i].split()) > 5:
                            self.pairs.append((texts[i], texts[i+1]))
                    self.vocab = vocab
                    self.max_len = max_len
                def __len__(self): return len(self.pairs)
                def __getitem__(self, idx):
                    src_txt, tgt_txt = self.pairs[idx]
                    src = torch.tensor(self.vocab.encode(src_txt)[:self.max_len])
                    tgt = torch.tensor(self.vocab.encode(tgt_txt)[:self.max_len])
                    # Padding manual strict
                    if len(src) < self.max_len: src = torch.cat([src, torch.zeros(self.max_len-len(src)).long()])
                    # Truncate if longer (in case encode didnt)
                    src = src[:self.max_len]
                    
                    if len(tgt) < self.max_len: tgt = torch.cat([tgt, torch.zeros(self.max_len-len(tgt)).long()])
                    tgt = tgt[:self.max_len]
                    
                    return src, tgt
            
            dataset = SimpleDataset(texts, vocab, mod.Config.MAX_LEN)

        dl = DataLoader(dataset, batch_size=16, shuffle=False)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        total_loss = 0
        start_time = time.time()
        
        model.eval()
        with torch.no_grad():
            for src, tgt in dl:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                
                if 'transf' in model_name.lower():
                    # Transformer Scratch espera (src, tgt_input)
                    # output shape (B, L, V) vs target (B, L)
                    out = model(src, tgt[:, :-1])
                    loss = criterion(out.reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
                else: 
                    # FastText LSTMs
                    out = model(src, tgt, 0)
                    loss = criterion(out[:, 1:].reshape(-1, vocab_size), tgt[:, 1:].reshape(-1))
                
                total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        ppl = np.exp(avg_loss)
        
        return {
            "Model": model_name,
            "Type": "Encoder-Decoder",
            "PPL": ppl,
            "Params (M)": count_parameters(model) / 1e6
        }
        
    except Exception as e:
        print(f"Error evaluando {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def eval_dec_only(mod_path, texts, model_name):
    """Evaluador para Decoder Only (Transformer y LSTM)"""
    print(f"Evaluando {model_name}...")
    try:
        mod = load_module(mod_path)
        
        # Vocabulario y Modelo
        vocab = mod.Vocabulary()
        
        # Unify vocab loading
        if hasattr(vocab, 'load'):
            vocab.load(mod.Config.VOCAB_PATH)
        elif hasattr(vocab, 'load_vocab'):
            vocab.load_vocab(mod.Config.VOCAB_PATH)
        else:
             print("Warning: Unknown vocab load method")
             
        if any(x in model_name.lower() for x in ['lstm', 'gru']):
            vocab_size = vocab.vocab_size
            checkpoint = torch.load(mod.Config.MODEL_PATH, map_location=DEVICE)
            if 'lstm' in model_name.lower():
                model = mod.LSTMGenerator(vocab_size, mod.Config.EMBEDDING_DIM, mod.Config.HIDDEN_DIM, 
                                        mod.Config.NUM_LAYERS, mod.Config.DROPOUT).to(DEVICE)
            else:
                model = mod.GRUGenerator(vocab_size, mod.Config.EMBEDDING_DIM, mod.Config.HIDDEN_DIM, 
                                       mod.Config.NUM_LAYERS, mod.Config.DROPOUT).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Transformer Generator
            vocab_size = len(vocab.token2idx)
            checkpoint = torch.load(mod.Config.MODEL_PATH, map_location=DEVICE)
            model = mod.TransformerGenerator(vocab_size, mod.Config.D_MODEL, mod.Config.N_HEADS, 
                                           mod.Config.N_LAYERS, mod.Config.D_FF, mod.Config.MAX_SEQ_LEN).to(DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

        model.eval()
        
        # Dataset
        # Usamos su propia clase TextDataset si es posible, o recreamos lógica simple de sliding window
        # Importante: TextDataset de GRU/LSTM espera un string gigante, no una lista de frases.
        if isinstance(texts, list):
            eval_text_full = " ".join(texts)
        else:
            eval_text_full = texts
            
        dataset = mod.TextDataset(eval_text_full, vocab, mod.Config.SEQ_LENGTH if hasattr(mod.Config, 'SEQ_LENGTH') else mod.Config.MAX_SEQ_LEN)
        dl = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # padding_idx es 0 para todos los modelos (Word-Level)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        total_loss = 0
        start_time = time.time()
        
        with torch.no_grad():
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                
                if any(x in model_name.lower() for x in ['lstm', 'gru']):
                    # LSTM/GRU return (logits, hidden)
                    output_tuple = model(x, None)
                    logits = output_tuple[0]
                    # Check shape
                    if logits.dim() == 3: # (B, L, V)
                         loss = criterion(logits.transpose(1, 2), y)
                    else:
                         loss = criterion(logits, y.view(-1))
                else: 
                    # Transformer Dec Only
                    logits = model(x)
                    if isinstance(logits, tuple): logits = logits[0]
                    
                    # Transformer entrena prediciendo shift
                    loss = criterion(logits.reshape(-1, vocab_size), y.view(-1))
                
                total_loss += loss.item()

        avg_loss = total_loss / len(dl)
        ppl = np.exp(avg_loss)
        
        # Todos los modelos generativos ahora son Word-Level verificado
        m_type = "Decoder-Only (Word)"
             
        return {
            "Model": model_name,
            "Type": m_type,
            "PPL": ppl,
            "Params (M)": count_parameters(model) / 1e6
        }
        
    except Exception as e:
        print(f"Error evaluando {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def eval_tinyllama(texts, model_name):
    """Evaluador para TinyLlama (PEFT/LoRA + 4-bit)"""
    print(f"Evaluando {model_name}...")
    if not HAS_PEFT:
        print("Saltando TinyLlama: peft no instalado")
        return None
        
    try:
        # Importar dinámicamente el módulo para obtener Config
        mod_path = os.path.join(MODELS_DEC_ONLY_DIR, 'tinyLlama_1_1B+QLoRA_4_bit.py')
        mod = load_module(mod_path)
        
        if not os.path.exists(mod.Config.ADAPTER_PATH):
             # Intentar cargar sin el adapter si no existe, o saltar
             print(f"Saltando {model_name}: No se encontró el adapter en {mod.Config.ADAPTER_PATH}")
             return None
            
        # Cargar Modelo (esto es pesado)
        # Seteamos hf_logging a error para evitar ruido
        hf_logging.set_verbosity_error()
        model, tokenizer = mod.load_finetuned_model()
        model.eval()
        
        # Para PPL necesitamos loss. HF CausalLM devuelve loss si pasamos labels.
        total_loss = 0
        samples = 0
        start_time = time.time()
        
        # Evaluamos una muestra significativa
        eval_texts = texts[:min(50, len(texts))]
        
        with torch.no_grad():
            for text in eval_texts:
                words = text.split()
                if len(words) < 5: continue
                
                split_point = min(len(words)//3, 10)
                start = ' '.join(words[:split_point])
                continuation = ' '.join(words[split_point:])
                
                # Usar el formato exacto del entrenamiento
                prompt = f"<|user|>\n{start}</s>\n<|assistant|>\n{continuation}</s>"
                
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item()
                samples += 1
                
        if samples == 0: return None
        
        avg_loss = total_loss / samples
        ppl = np.exp(avg_loss)
        
        # Limpiar memoria GPU (importante con LLMs)
        del model
        torch.cuda.empty_cache()
        
        return {
            "Model": model_name,
            "Type": "LLM (4-bit)",
            "PPL": ppl,
            "Params (M)": 1100.0, 
        }
        
    except Exception as e:
        print(f"Error evaluando {model_name}: {e}")
        return None

# ==================================================================================
# MAIN
# ==================================================================================

def main():
    val_texts = get_validation_data()
    print(f"Dataset validación cargado: {len(val_texts)} frases.")
    
    results = []
    
    # 1. ENCODER-DECODER BERT
    results.append(eval_enc_dec_bert(os.path.join(MODELS_ENC_DEC_DIR, 'lstm_bert.py'), val_texts, 'LSTM+BERT'))
    results.append(eval_enc_dec_bert(os.path.join(MODELS_ENC_DEC_DIR, 'gru_bert.py'), val_texts, 'GRU+BERT'))
    results.append(eval_enc_dec_bert(os.path.join(MODELS_ENC_DEC_DIR, 'transformer_bert.py'), val_texts, 'Transf+BERT'))

    # 2. ENCODER-DECODER SCRATCH / FASTTEXT
    results.append(eval_enc_dec_scratch(os.path.join(MODELS_ENC_DEC_DIR, 'transformer_scratch.py'), val_texts, 'Transf Scratch'))
    results.append(eval_enc_dec_scratch(os.path.join(MODELS_ENC_DEC_DIR, 'lstm_fasttext.py'), val_texts, 'LSTM+FastText'))
    results.append(eval_enc_dec_scratch(os.path.join(MODELS_ENC_DEC_DIR, 'gru_fasttext.py'), val_texts, 'GRU+FastText'))
    
    # 3. DECODER ONLY
    results.append(eval_dec_only(os.path.join(MODELS_DEC_ONLY_DIR, 'transformer_generator.py'), val_texts, 'Transf Generator'))
    results.append(eval_dec_only(os.path.join(MODELS_DEC_ONLY_DIR, 'lstm_generator.py'), val_texts, 'LSTM Generator'))
    results.append(eval_dec_only(os.path.join(MODELS_DEC_ONLY_DIR, 'gru_generator.py'), val_texts, 'GRU Generator'))
    
    # 4. LLM
    results.append(eval_tinyllama(val_texts, 'TinyLlama 1.1B'))
    
    # Filtrar fallos
    results = [r for r in results if r is not None]
    df_res = pd.DataFrame(results)
    
    print("\nRESULTADOS:")
    print(df_res)
    
    # ==================== GRAFICOS ====================
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Perplexity (Barplot separado por tipo)
    sns.barplot(data=df_res, x='Model', y='PPL', hue='Type', ax=axes[0])
    axes[0].set_title('Perplexity (Menor es mejor)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_yscale('log') # PPL puede variar mucho
    
    # 2. Parámetros vs PPL
    sns.scatterplot(data=df_res, x='Params (M)', y='PPL', hue='Type', style='Model', s=200, ax=axes[1])
    axes[1].set_title('Eficiencia: Tamaño vs Calidad')
    axes[1].set_yscale('log')
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'evaluacion_modelos.png'))
    print(f"\nGráficos guardados en {os.path.join(IMG_DIR, 'evaluacion_modelos.png')}")

if __name__ == "__main__":
    main()
