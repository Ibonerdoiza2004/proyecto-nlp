# ============================================================
# 0) Instalación (ejecutad esto una vez en el entorno)
# ============================================================
# !pip install pandas numpy spacy==3.* gensim scikit-learn transformers torch joblib unidecode
# !python -m spacy download es_core_news_md  # o es_core_news_sm si hay pocas RAM

# ============================================================
# 1) Carga de datos
# ============================================================
import re
import string
import joblib
import numpy as np
import pandas as pd
from unidecode import unidecode

import spacy
from spacy.lang.es.stop_words import STOP_WORDS as SPACY_STOP_ES

df = pd.read_csv("./dataset/dataset_unificado.csv")  # columnas: audio_id, start_sec, end_sec, duration_sec, speaker, text,...

# ============================================================
# 2) Limpieza ligera (respetando la información lingüística)
#    - En atribución de hablante evitamos "sobre-limpiar"
# ============================================================
MULTISPACE = re.compile(r"\s+")

def basic_clean(text: str,
                lower=True,
                normalize_quotes=True,
                strip_accents=False):
    if not isinstance(text, str):
        return ""

    # normalización ligera de comillas y espacios
    if normalize_quotes:
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    # bajar a minúsculas (ojo: pierde info de nombres propios; aceptable aquí)
    if lower:
        text = text.lower()

    # (opcional) quitar acentos para modelos basados en bolsa de palabras
    if strip_accents:
        text = unidecode(text)

    # compactar espacios
    text = MULTISPACE.sub(" ", text).strip()
    return text

df["text_clean"] = df["text"].apply(basic_clean)

# ============================================================
# 3) Tokenización + Lemas + POS con spaCy (es_core_news_md)
# ============================================================

import spacy
from spacy.lang.es.stop_words import STOP_WORDS as SPACY_STOP_ES

nlp = spacy.load("es_core_news_md", disable=["ner"])
CUSTOM_STOP = set(SPACY_STOP_ES) - {"no", "ni"}

def process_docs(texts, batch_size=256):
    rows = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = [t.text for t in doc if not t.is_space]
        lemmas = [t.lemma_ for t in doc if not t.is_space]
        pos    = [t.pos_ for t in doc if not t.is_space]
        # usa t.is_punct (cubre ¿, ¡, …, etc.)
        tokens_no_stop = [t.text for t in doc
                          if not t.is_space and not t.is_punct
                          and t.text not in CUSTOM_STOP and t.lemma_ not in CUSTOM_STOP]
        lemmas_no_stop = [t.lemma_ for t in doc
                          if not t.is_space and not t.is_punct
                          and t.text not in CUSTOM_STOP and t.lemma_ not in CUSTOM_STOP]
        rows.append((tokens, lemmas, pos, tokens_no_stop, lemmas_no_stop))
    return rows

proc_rows = process_docs(df["text_clean"].tolist(), batch_size=256)
df[["tokens", "lemmas", "pos", "tokens_no_stop", "lemmas_no_stop"]] = pd.DataFrame(proc_rows, index=df.index)

# ============================================================
# 5) n-grams de palabras (bigrams/trigrams) con scikit-learn
#    - útil para capturar expresiones tipo "juego de", "liga española"
# ============================================================
from sklearn.feature_extraction.text import CountVectorizer

# texts ya limpios; construimos para tokens sin stopwords
texts_for_ngrams = df["lemmas_no_stop"].apply(lambda xs: " ".join(xs)).tolist()

# BOW de uni+bi+trigramas (palabras)
bow_vectorizer = CountVectorizer(ngram_range=(1,3), min_df=5)  # ajustad min_df según corpus
X_bow = bow_vectorizer.fit_transform(texts_for_ngrams)

# ============================================================
# 6) TF-IDF (palabras y, si queréis, caracteres)
# ============================================================
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_word = TfidfVectorizer(ngram_range=(1,2), min_df=5)
X_tfidf_word = tfidf_word.fit_transform(texts_for_ngrams)

# (opcional) TF-IDF a nivel de caracteres — captura estilo (sufijos, tildes, signos) útil en autoría
tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=5)
X_tfidf_char = tfidf_char.fit_transform(df["text_clean"].tolist())

# ============================================================
# 7) Word Embeddings no contextuales (Word2Vec / FastText entrenados en vuestro corpus)
#    Si tenéis preentrenados, cargadlos; si no, entrenamos para cubrir el dominio.
# ============================================================
from gensim.models import Word2Vec, FastText

sentences = df["lemmas_no_stop"].tolist()

w2v = Word2Vec(
    sentences=sentences,
    vector_size=200, window=5, min_count=5, workers=4, sg=1, epochs=10
)

ft = FastText(
    sentences=sentences,
    vector_size=200, window=5, min_count=3, workers=4, sg=1, epochs=10
)

# Función para "embeddings de documento" promediando palabras (simple, rápido)
def average_word_vectors(tokens, keyed_vectors, missing_token="__OOV__"):
    vecs = []
    for t in tokens:
        if t in keyed_vectors.key_to_index:
            vecs.append(keyed_vectors[t])
    if not vecs:  # si todas OOV, devolvemos vector nulo
        return np.zeros(keyed_vectors.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)

df["w2v_avg"] = df["lemmas_no_stop"].apply(lambda toks: average_word_vectors(toks, w2v.wv))
df["ft_avg"]  = df["lemmas_no_stop"].apply(lambda toks: average_word_vectors(toks, ft.wv))

# ============================================================
# 8) Embeddings contextuales (BETO/BERT en español)
#    - Representación frase/documento; se recomienda para autoría y estilo
# ============================================================
import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"  # BETO
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
            pooled = out.last_hidden_state[:, 0]  # [CLS]
        else:
            # mean pooling con máscara de atención
            last_hidden = out.last_hidden_state
            mask = batch_enc["attention_mask"].unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
        embs.append(pooled.cpu().numpy())
    return np.vstack(embs)

# Calculad cuando lo necesitéis (tarda). Aquí mostramos cómo:
df["bert_cls"]  = list(bert_embed(df["text_clean"].tolist(), pooling="cls"))
df["bert_mean"] = list(bert_embed(df["text_clean"].tolist(), pooling="mean"))

# ============================================================
# 9) Guardado de artefactos para reutilizar en modelado
# ============================================================
# a) Datos enriquecidos
df_out = df.copy()
# Convertir listas a strings JSON-compat si vais a CSV; mejor usar parquet para mantener listas
df_out.to_csv("dataset_preprocesado.csv", index=False)

# b) Vectorizadores y modelos para reusar en la E3
joblib.dump(bow_vectorizer, "vec_bow.joblib")
joblib.dump(tfidf_word,   "vec_tfidf_word.joblib")
joblib.dump(tfidf_char,   "vec_tfidf_char.joblib")
w2v.save("w2v.model")
ft.save("fasttext.model")

print("Listo: dataset_preprocesado.parquet + vectorizadores y modelos guardados.")
