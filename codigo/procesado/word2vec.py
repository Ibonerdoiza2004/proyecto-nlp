import ast
import numpy as np
import pandas as pd
import pickle
from gensim.models import Word2Vec, FastText

# Leer el dataset preprocesado
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

# --- NUEVO: Generar y guardar vocabulario común ---
print("Generando vocabulario común...")
all_words = [word for text in sentences for word in text]
# Usamos sorted() para garantizar que el orden sea siempre el mismo (determinista)
vocab = sorted(list(set(all_words))) 

word2idx = {word: idx+2 for idx, word in enumerate(vocab)}
word2idx['<pad>'] = 0
word2idx['<unk>'] = 1

# Guardar el vocabulario para usarlo en entrenamiento y análisis
with open("models/word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)
print(f"Vocabulario guardado en models/word2idx.pkl (Tamaño: {len(word2idx)})")
# --------------------------------------------------

# Entrenamiento
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

# Guardadr csvs y modelos
df_out = df[["audio_id", "start_sec", "end_sec", "duration_sec", "speaker", "text", "text_clean", "w2v_avg", "ft_avg"]].copy()
df_out.to_csv("dataset/dataset_w2v_ft.csv", index=False)

w2v.save("models/w2v.model")
ft.save("models/fasttext.model")

print("Guardado: dataset_w2v_ft.csv, w2v.model, fasttext.model")
