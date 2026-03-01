# Proyecto NLP — Análisis de Podcasts de Fútbol en Español

Proyecto de Procesamiento del Lenguaje Natural (NLP) para el análisis de podcasts de fútbol en español. El proyecto abarca desde la recolección de datos de audio hasta el entrenamiento de modelos de aprendizaje automático para **clasificación de hablante** y **generación de texto**.

---

## Estructura del proyecto

```
proyecto-nlp/
├── codigo/
│   ├── preparar_dataset/       # Descarga, transcripción y diarización de audio
│   │   ├── download_mp3.py         # Descarga vídeos de YouTube como MP3
│   │   ├── transcript.py           # Transcripción de audio con Whisper
│   │   ├── diarization.py          # Diarización de hablantes con pyannote.audio
│   │   ├── unificar_dataset.py     # Unifica transcripción y diarización en un CSV
│   │   └── fusionar_media.py       # Fusiona archivos de audio/vídeo
│   ├── procesado/              # Preprocesamiento y representaciones de texto
│   │   ├── preprocesado.py         # Limpieza, tokenización, lematización (spaCy)
│   │   ├── rep_tradicional.py      # Representaciones BoW y TF-IDF
│   │   ├── word2vec.py             # Embeddings Word2Vec y FastText
│   │   └── BETO.py                 # Embeddings con BETO (BERT en español)
│   ├── modelos ML/
│   │   ├── clasificacion hablante/ # Modelos para identificar al hablante
│   │   │   ├── shallow/            # Modelos ML clásicos (SVM, RF, LR, NB, DT)
│   │   │   ├── perceptron/         # Perceptrón multicapa (MLP)
│   │   │   ├── cnn/                # Redes neuronales convolucionales
│   │   │   ├── lstm/               # Redes LSTM
│   │   │   ├── gru/                # Redes GRU
│   │   │   └── transformers/       # Clasificadores basados en BERT
│   │   └── generacion texto/       # Modelos para generación de texto
│   │       ├── decoder_only/       # LSTM, GRU, Transformer, TinyLlama + QLoRA
│   │       └── encoder_decoder/    # Arquitecturas encoder-decoder con BERT/FastText
│   └── analisis/               # Evaluación y análisis de resultados
│       ├── analisis_estadistico.py
│       ├── analisisModelosML.py
│       ├── analisisModelosML_generacion.py
│       ├── analizar_rep_tradicional.py
│       ├── analizar_word2vec.py
│       ├── analizarBETO.py
│       └── evaluar_generacion.py
├── dataset/
│   ├── dataset_unificado.csv       # Dataset bruto con segmentos de audio etiquetados
│   └── dataset_preprocesado.csv    # Dataset tras el preprocesamiento NLP
├── imagenes/                   # Gráficas de entrenamiento y matrices de confusión
├── graficos/                   # Estadísticas del dataset (duración, palabras, etc.)
├── transcripts/                # Transcripciones en formato JSON/SRT por audio
└── diarizado/                  # Resultados de diarización por audio
```

---

## Pipeline

```
YouTube (MP3)
    │
    ▼
Transcripción (Whisper)
    │
    ▼
Diarización de hablantes (pyannote.audio)
    │
    ▼
Dataset unificado (CSV)
    │
    ▼
Preprocesamiento NLP (spaCy)
    │
    ▼
Representaciones de texto
  ├── BoW / TF-IDF
  ├── Word2Vec / FastText
  └── BETO (dccuchile/bert-base-spanish-wwm-cased)
    │
    ▼
Modelos ML
  ├── Clasificación de hablante
  └── Generación de texto
```

---

## Tareas

### 1. Clasificación de hablante

Dado un segmento de texto, el modelo predice qué hablante lo pronunció.

Representaciones usadas como entrada: **BoW**, **TF-IDF** (palabras y caracteres), **Word2Vec**, **FastText**, **BETO CLS**.

| Familia de modelos | Variantes |
|---|---|
| Shallow ML | Logistic Regression, SVM, Random Forest, Decision Tree, Naive Bayes |
| Perceptrón (MLP) | BoW, TF-IDF (palabras/caracteres), BETO CLS |
| CNN | Word2Vec, FastText, BERT |
| LSTM / BiLSTM | Word2Vec, FastText, BERT |
| GRU / BiGRU | Word2Vec, FastText, BERT |
| Transformers | BERT con atención, AutoModel |

### 2. Generación de texto

Modelos entrenados para generar texto al estilo de los hablantes del podcast.

| Arquitectura | Modelos |
|---|---|
| Decoder-only | LSTM, GRU, Transformer from scratch, TinyLlama-1.1B + QLoRA (4-bit) |
| Encoder-Decoder | Transformer, LSTM + BERT/FastText, GRU + BERT/FastText |

---

## Requisitos principales

- Python 3.9+
- [PyTorch](https://pytorch.org/)
- [Transformers (Hugging Face)](https://huggingface.co/docs/transformers)
- [spaCy](https://spacy.io/) + modelo `es_core_news_md`
- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [gensim](https://radimrehurek.com/gensim/) (Word2Vec / FastText)
- [scikit-learn](https://scikit-learn.org/)
- [PEFT](https://github.com/huggingface/peft) + [TRL](https://github.com/huggingface/trl) (para QLoRA)
- pandas, numpy, matplotlib, seaborn

---

## Uso

### 1. Preparar el dataset

```bash
# Descargar audios de YouTube
python codigo/preparar_dataset/download_mp3.py

# Transcribir con Whisper
python codigo/preparar_dataset/transcript.py

# Diarizar hablantes
python codigo/preparar_dataset/diarization.py

# Unificar en un único CSV
python codigo/preparar_dataset/unificar_dataset.py
```

### 2. Preprocesar texto

```bash
python codigo/procesado/preprocesado.py
```

### 3. Generar representaciones

```bash
python codigo/procesado/rep_tradicional.py  # BoW y TF-IDF
python codigo/procesado/word2vec.py          # Word2Vec y FastText
python codigo/procesado/BETO.py              # Embeddings BETO
```

### 4. Entrenar modelos de clasificación de hablante

```bash
# Ejemplo: Shallow ML con TF-IDF
python "codigo/modelos ML/clasificacion hablante/shallow/shallowML_TFIDF.py"

# Ejemplo: LSTM con Word2Vec
python "codigo/modelos ML/clasificacion hablante/lstm/lstmWord2Vec.py"

# Ejemplo: BERT con atención
python "codigo/modelos ML/clasificacion hablante/transformers/bert_attention_classifier.py"
```

### 5. Entrenar modelos de generación de texto

```bash
# Ejemplo: GRU decoder-only
python "codigo/modelos ML/generacion texto/decoder_only/gru_generator.py"

# Ejemplo: TinyLlama + QLoRA
python "codigo/modelos ML/generacion texto/decoder_only/tinyLlama_1_1B+QLoRA_4_bit.py"
```

### 6. Analizar resultados

```bash
python codigo/analisis/analisisModelosML.py
python codigo/analisis/analisisModelosML_generacion.py
python codigo/analisis/analisis_estadistico.py
```

---

## Dataset

El dataset `dataset_unificado.csv` contiene segmentos de audio etiquetados con el hablante y la transcripción, con las siguientes columnas:

| Columna | Descripción |
|---|---|
| `audio_id` | Identificador del vídeo de YouTube |
| `start_sec` | Segundo de inicio del segmento |
| `end_sec` | Segundo de fin del segmento |
| `duration_sec` | Duración del segmento en segundos |
| `speaker` | Etiqueta del hablante (p.ej., MIGUEL, NAHUEL) |
| `text` | Transcripción del segmento |
| `n_chars` | Número de caracteres |
| `n_words` | Número de palabras |

Tras el preprocesamiento (`dataset_preprocesado.csv`) se añaden columnas adicionales: `text_clean`, `tokens`, `lemmas`, `pos`, `tokens_no_stop`, `lemmas_no_stop`.
