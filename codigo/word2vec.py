"""
Embeddings no contextuales:
- Lee dataset_preprocesado.csv
- Entrena Word2Vec y FastText (mismos hiperparámetros que pasaste)
- Calcula promedios por documento
- Guarda dataset_w2v_ft.csv con columnas w2v_avg y ft_avg (listas)
- Guarda modelos w2v.model y fasttext.model
"""

import ast
import numpy as np
import pandas as pd
from gensim.models import Word2Vec, FastText

# Leer el preprocesado
df = pd.read_csv("dataset/dataset_preprocesado.csv")

def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)
sentences = df["lemmas_no_stop"].tolist()

# Entrenamiento (idéntico a tu código)
w2v = Word2Vec(
    sentences=sentences,
    vector_size=200, window=5, min_count=5, workers=4, sg=1, epochs=10
)

ft = FastText(
    sentences=sentences,
    vector_size=200, window=5, min_count=3, workers=4, sg=1, epochs=10
)

def average_word_vectors(tokens, keyed_vectors):
    vecs = []
    for t in tokens:
        if t in keyed_vectors.key_to_index:
            vecs.append(keyed_vectors[t])
    if not vecs:
        return np.zeros(keyed_vectors.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)

df["w2v_avg"] = df["lemmas_no_stop"].apply(lambda toks: average_word_vectors(toks, w2v.wv).tolist())
df["ft_avg"]  = df["lemmas_no_stop"].apply(lambda toks: average_word_vectors(toks, ft.wv).tolist())

# Guardados
df_out = df[["audio_id", "start_sec", "end_sec", "duration_sec", "speaker", "text", "text_clean", "w2v_avg", "ft_avg"]].copy()
df_out.to_csv("dataset/dataset_w2v_ft.csv", index=False)

w2v.save("models/w2v.model")
ft.save("models/fasttext.model")

print("Guardado: dataset_w2v_ft.csv, w2v.model, fasttext.model")
