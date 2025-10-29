"""
Embeddings contextuales (BETO):
- Lee dataset_preprocesado.csv
- Calcula embeddings con BETO (CLS y mean pooling)
- Guarda dataset_bert.csv con columnas bert_cls y bert_mean (listas)
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Leer el preprocesado
df = pd.read_csv("dataset/dataset_preprocesado.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
bert = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

@torch.no_grad()
def bert_embed(texts, batch_size=16, max_length=256, pooling="cls"):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_enc = tok(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(DEVICE)
        out = bert(**batch_enc)
        if pooling == "cls":
            pooled = out.last_hidden_state[:, 0]
        else:
            last_hidden = out.last_hidden_state
            mask = batch_enc["attention_mask"].unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        embs.append(pooled.cpu().numpy())
    return np.vstack(embs)

texts = df["text_clean"].tolist()

# === Calcula embeddings (dos variantes) ===
emb_cls  = bert_embed(texts, pooling="cls").astype("float32")
emb_mean = bert_embed(texts, pooling="mean").astype("float32")

# === Guarda en .npz para análisis rápido ===
np.savez_compressed("models/bert_cls.npz",  emb_cls)
np.savez_compressed("models/bert_mean.npz", emb_mean)

df["bert_cls"]  = list(bert_embed(texts, pooling="cls"))
df["bert_mean"] = list(bert_embed(texts, pooling="mean"))

# Convertir a listas para CSV
df["bert_cls"]  = df["bert_cls"].apply(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)
df["bert_mean"] = df["bert_mean"].apply(lambda v: v.tolist() if isinstance(v, np.ndarray) else v)

df_out = df[["audio_id", "start_sec", "end_sec", "duration_sec", "speaker", "text", "text_clean", "bert_cls", "bert_mean"]].copy()
df_out.to_csv("dataset/dataset_bert.csv", index=False)


print("Guardado: dataset_bert.csv")
