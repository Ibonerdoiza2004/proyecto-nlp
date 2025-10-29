import ast
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

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

# Textos para n-gramas
texts_for_ngrams = df["lemmas_no_stop"].apply(lambda xs: " ".join(xs)).tolist()

# BoW 1-3
bow_vectorizer = CountVectorizer(ngram_range=(1,3), min_df=5)
X_bow = bow_vectorizer.fit_transform(texts_for_ngrams)

# TF-IDF de palabras 1-2
tfidf_word = TfidfVectorizer(ngram_range=(1,2), min_df=5)
X_tfidf_word = tfidf_word.fit_transform(texts_for_ngrams)

# TF-IDF de caracteres 3-5
tfidf_char = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=5)
X_tfidf_char = tfidf_char.fit_transform(df["text_clean"].tolist())

# Guardar vectorizadores
joblib.dump(bow_vectorizer, "models/vec_bow.joblib")
joblib.dump(tfidf_word,   "models/vec_tfidf_word.joblib")
joblib.dump(tfidf_char,   "models/vec_tfidf_char.joblib")

print("Listo: vec_bow.joblib, vec_tfidf_word.joblib, vec_tfidf_char.joblib")