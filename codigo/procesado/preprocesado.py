import re
import pandas as pd
from unidecode import unidecode

import spacy
from spacy.lang.es.stop_words import STOP_WORDS as SPACY_STOP_ES

df = pd.read_csv("./dataset/dataset_unificado.csv")  # columnas: audio_id, start_sec, end_sec, duration_sec, speaker, text,...

# Limpieza de datos
MULTISPACE = re.compile(r"\s+")

def basic_clean(text: str,
                lower=True,
                normalize_quotes=True,
                strip_accents=False):
    if not isinstance(text, str):
        return ""

    if normalize_quotes:
        text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    if lower:
        text = text.lower()

    if strip_accents:
        text = unidecode(text)

    text = MULTISPACE.sub(" ", text).strip()
    return text

df["text_clean"] = df["text"].apply(basic_clean)

# 3) Tokenización + Lemas + POS con spaCy
nlp = spacy.load("es_core_news_md", disable=["ner"])
CUSTOM_STOP = set(SPACY_STOP_ES) - {"no", "ni"}

def process_docs(texts, batch_size=256):
    rows = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens = [t.text for t in doc if not t.is_space]
        lemmas = [t.lemma_ for t in doc if not t.is_space]
        pos    = [t.pos_ for t in doc if not t.is_space]
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


# Guardado de csv
df.to_csv("dataset/dataset_preprocesado.csv", index=False)

print("Guardado: dataset_preprocesado.csv")