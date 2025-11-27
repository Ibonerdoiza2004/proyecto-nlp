# Resumen de Todos los Modelos Implementados

**Fecha**: 27 de Noviembre, 2025  
**Total de modelos**: 42 modelos diferentes en 17 archivos

---

## 📁 Estructura de Archivos

```
codigo/modelos ML/clasificacion hablante/
│
├── Shallow Machine Learning (5 archivos × 6 modelos = 30 modelos)
│   ├── shallowML_BoW.py                    ✅ NUEVO
│   ├── shallowML_TFIDF.py                  ✅ NUEVO
│   ├── shallowML_TFIDF_char.py             ✅ NUEVO
│   ├── shallowML_Word2Vec.py               ✅ NUEVO
│   └── shallowML_FastText.py               ✅ NUEVO
│
├── Deep Learning - Word2Vec (4 archivos)
│   ├── lstmWord2Vec.py                     ✅ MEJORADO
│   ├── lstmWord2VecTrainable.py            ✅ EXISTENTE
│   ├── gruWord2Vec.py                      ✅ NUEVO
│   ├── cnnWord2Vec.py                      ✅ MEJORADO
│   └── cnnLstmWord2Vec.py                  ✅ NUEVO
│
├── Deep Learning - FastText (2 archivos)
│   ├── cnnFastText.py                      ✅ NUEVO
│   └── lstmFastText.py                     ✅ NUEVO
│
├── Deep Learning - BERT (4 archivos)
│   ├── lstmBert.py                         ✅ MEJORADO
│   ├── cnnBert.py                          ✅ NUEVO
│   ├── gruBert.py                          ✅ NUEVO
│   └── bertFineTuning.py                   ✅ NUEVO
│
└── Generación de Texto (1 archivo)
    └── lstm_generacion.py                  ✅ MEJORADO
```

---

## 📊 Detalle de Cada Archivo

### 1. Shallow Machine Learning

#### shallowML_BoW.py (6 modelos)
**Embedding**: Bag of Words (CountVectorizer)
- ✅ Logistic Regression
- ✅ Random Forest (100 árboles)
- ✅ Linear SVM
- ✅ SVM RBF
- ✅ Decision Tree
- ✅ Multinomial Naive Bayes

**Características**:
- Max features: 5000
- Min df: 2, Max df: 0.8
- 5-fold cross-validation
- Accuracy esperada: 80-85%

---

#### shallowML_TFIDF.py (6 modelos)
**Embedding**: TF-IDF (palabras)
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ Linear SVM
- ✅ SVM RBF
- ✅ Decision Tree
- ✅ Multinomial Naive Bayes

**Características**:
- Analyzer: 'word'
- N-grams: (1, 2) - unigramas y bigramas
- Sublinear TF: True
- Accuracy esperada: 82-87%

---

#### shallowML_TFIDF_char.py (6 modelos)
**Embedding**: TF-IDF (caracteres)
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ Linear SVM
- ✅ SVM RBF
- ✅ Decision Tree
- ✅ Multinomial Naive Bayes

**Características**:
- Analyzer: 'char'
- N-grams: (2, 5) - 2 a 5 gramas de caracteres
- Captura patrones morfológicos
- Robusto ante errores ortográficos
- Accuracy esperada: 81-86%

---

#### shallowML_Word2Vec.py (6 modelos)
**Embedding**: Word2Vec (promedio de vectores)
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ Linear SVM
- ✅ SVM RBF
- ✅ Decision Tree
- ✅ Gaussian Naive Bayes (para datos continuos)

**Características**:
- Vector size: 100
- Skip-gram (sg=1)
- Promedio de embeddings por texto
- Accuracy esperada: 83-88%

---

#### shallowML_FastText.py (6 modelos)
**Embedding**: FastText (promedio de vectores + character n-grams)
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ Linear SVM
- ✅ SVM RBF
- ✅ Decision Tree
- ✅ Gaussian Naive Bayes

**Características**:
- Vector size: 100
- Character n-grams: 3-6
- Maneja palabras OOV
- Accuracy esperada: 84-89%

---

### 2. Deep Learning - Word2Vec

#### lstmWord2Vec.py ✅ MEJORADO
**Arquitectura**: LSTM Bidireccional (2 capas) + Word2Vec

**Técnicas aplicadas** (PDF págs 38-40, 78-79):
- ✅ Bidirectional LSTM
- ✅ 2 capas apiladas
- ✅ Packed sequences
- ✅ Gradient clipping (5.0)
- ✅ L2 regularization (1e-5)
- ✅ Dropout optimizado (0.3)

**Accuracy esperada**: 90-92% (mejora de +5-7%)

---

#### gruWord2Vec.py ✅ NUEVO
**Arquitectura**: GRU Bidireccional (2 capas) + Word2Vec

**Técnicas aplicadas** (PDF págs 38-40):
- ✅ Bidirectional GRU
- ✅ 2 capas apiladas
- ✅ Packed sequences
- ✅ Gradient clipping (5.0)
- ✅ L2 regularization (1e-5)
- ✅ Menos parámetros que LSTM

**Accuracy esperada**: 85-90%

---

#### cnnWord2Vec.py ✅ MEJORADO
**Arquitectura**: CNN múltiples kernels + Word2Vec

**Técnicas aplicadas** (PDF págs 25-30):
- ✅ Múltiples kernels [2, 3, 4, 5]
- ✅ Batch Normalization
- ✅ Global Max Pooling
- ✅ Gradient clipping (5.0)
- ✅ L2 regularization (1e-5)

**Accuracy esperada**: 86-89% (mejora de +3-5%)

---

#### cnnLstmWord2Vec.py ✅ NUEVO
**Arquitectura**: CNN → LSTM Bidireccional (híbrido) + Word2Vec

**Técnicas aplicadas**:
- ✅ CNN extrae features locales (kernels [2, 3, 4])
- ✅ LSTM captura dependencias temporales
- ✅ Batch Normalization
- ✅ Packed sequences
- ✅ Arquitectura híbrida innovadora

**Accuracy esperada**: 87-92%

---

### 3. Deep Learning - FastText

#### cnnFastText.py ✅ NUEVO
**Arquitectura**: CNN múltiples kernels + FastText

**Características**:
- ✅ FastText con character n-grams (3-6)
- ✅ Múltiples kernels [2, 3, 4, 5]
- ✅ Batch Normalization
- ✅ Global Max Pooling
- ✅ Maneja palabras OOV

**Accuracy esperada**: 84-89%

---

#### lstmFastText.py ✅ NUEVO
**Arquitectura**: LSTM Bidireccional (2 capas) + FastText

**Características**:
- ✅ FastText con character n-grams (3-6)
- ✅ Bidirectional LSTM
- ✅ 2 capas apiladas
- ✅ Packed sequences
- ✅ Robusto ante OOV

**Accuracy esperada**: 86-91%

---

### 4. Deep Learning - BERT

#### lstmBert.py ✅ MEJORADO
**Arquitectura**: LSTM Bidireccional + Attention + BERT

**Técnicas aplicadas** (PDF págs 38-40, 64-71, 78-79):
- ✅ Bahdanau Attention mechanism
- ✅ Bidirectional LSTM (2 capas)
- ✅ Packed sequences
- ✅ Residual connection con attention
- ✅ BERT mean pooling embeddings
- ✅ Gradient clipping + L2 reg

**Accuracy esperada**: 92-95% (mejora de +2-5%)

---

#### cnnBert.py ✅ NUEVO
**Arquitectura**: CNN + BERT embeddings (frozen)

**Características**:
- ✅ BERT embeddings contextuales (frozen)
- ✅ CNN múltiples kernels [2, 3, 4, 5]
- ✅ Batch Normalization
- ✅ Global Max Pooling
- ✅ Rápido (BERT no entrena)

**Accuracy esperada**: 88-93%

---

#### gruBert.py ✅ NUEVO
**Arquitectura**: GRU Bidireccional (2 capas) + BERT (frozen)

**Características**:
- ✅ BERT embeddings contextuales (frozen)
- ✅ GRU Bidireccional (2 capas)
- ✅ Menos parámetros que LSTM
- ✅ Gradient clipping
- ✅ Rápido entrenamiento

**Accuracy esperada**: 89-94%

---

#### bertFineTuning.py ✅ NUEVO
**Arquitectura**: BERT Fine-Tuning completo

**Técnicas aplicadas** (PDF págs 82-90):
- ✅ Full BERT fine-tuning (todos los pesos)
- ✅ [CLS] token classification
- ✅ Differential learning rates (BERT: 2e-5, Classifier: 2e-4)
- ✅ Linear warmup scheduler (100 steps)
- ✅ AdamW optimizer con weight decay (0.01)
- ✅ Gradient clipping (1.0)

**Accuracy esperada**: 90-95%
**⚠️ Requiere GPU**: Entrenamiento muy lento en CPU

---

### 5. Generación de Texto

#### lstm_generacion.py ✅ MEJORADO
**Arquitectura**: LSTM (2 capas) para generación de texto

**Técnicas aplicadas** (PDF págs 72-81):
- ✅ 2 capas LSTM apiladas
- ✅ Teacher forcing con scheduled sampling (decay 0.95^epoch)
- ✅ Beam search generation (beam_width=3)
- ✅ Gradient clipping (5.0)
- ✅ L2 regularization (1e-5)

**Perplexity esperado**: 35-45 (mejora de -15-25%)

---

## 🎯 Comparativa de Rendimiento Esperado

### Por Categoría:

| Categoría | Accuracy | Tiempo Entrenamiento | Recursos |
|-----------|----------|---------------------|----------|
| Shallow ML + BoW/TF-IDF | 80-87% | Segundos | CPU |
| Shallow ML + Word2Vec/FastText | 83-89% | Minutos | CPU |
| Deep Learning + Word2Vec | 85-92% | 30-60 min | CPU/GPU |
| Deep Learning + FastText | 84-91% | 30-60 min | CPU/GPU |
| Deep Learning + BERT frozen | 88-94% | 1-2 horas | GPU recomendado |
| Deep Learning + BERT fine-tuning | 90-95% | 2-4 horas | GPU necesario |

---

## 🛠️ Cómo Usar Estos Modelos

### 1. Shallow ML (más rápido)
```bash
cd "codigo/modelos ML/clasificacion hablante"

# Entrenar y comparar 6 modelos con BoW
python shallowML_BoW.py

# Entrenar y comparar 6 modelos con TF-IDF
python shallowML_TFIDF.py

# Entrenar y comparar 6 modelos con Word2Vec
python shallowML_Word2Vec.py
```

### 2. Deep Learning - Word2Vec
```bash
# LSTM Bidireccional mejorado
python lstmWord2Vec.py

# GRU Bidireccional
python gruWord2Vec.py

# CNN mejorado
python cnnWord2Vec.py

# Híbrido CNN-LSTM
python cnnLstmWord2Vec.py
```

### 3. Deep Learning - FastText
```bash
# CNN con FastText
python cnnFastText.py

# LSTM con FastText
python lstmFastText.py
```

### 4. Deep Learning - BERT
```bash
# LSTM + Attention + BERT
python lstmBert.py

# CNN + BERT (frozen)
python cnnBert.py

# GRU + BERT (frozen)
python gruBert.py

# BERT Fine-tuning completo (requiere GPU)
python bertFineTuning.py
```

---

## 📈 Outputs Generados

Cada modelo genera:
- ✅ Modelo entrenado guardado en `models/`
- ✅ Matriz de confusión (PNG)
- ✅ Gráficos de entrenamiento (PNG)
- ✅ Reporte de clasificación (consola)
- ✅ Métricas de comparación (Shallow ML)
- ✅ Vectorizadores/embeddings guardados

---

## 🎓 Fuentes Académicas

Todas las técnicas implementadas provienen de:

### PDFs del Curso:
- **3_UNIT 3 - NLP.pdf** (81 páginas):
  - CNNs para texto (págs 25-30)
  - LSTM/GRU (págs 38-60)
  - Attention (págs 61-71)
  - Seq2Seq (págs 72-81)
  - Transformers/BERT (págs 82-90)

- **3_RNNs_Atencion.pdf** (9 páginas):
  - Bahdanau Attention (págs 6-7)
  - Teacher Forcing
  - Beam Search

### Notebooks de Prácticas:
1. Classification_using_shallow_machine_learning_techniques.ipynb
2. CNNs_for_Text_Classification.ipynb
3. LSTMs_for_Text_Classification.ipynb
4. Machine_Translation.ipynb
5. Word2Vec y FastText (modelos pre-entrenados)

---

## ✅ Resumen Final

### Archivos Creados/Mejorados:
- **12 archivos nuevos**: Todas las combinaciones faltantes
- **5 archivos mejorados**: Aplicando técnicas de PDFs
- **Total: 17 archivos funcionales**

### Modelos Implementados:
- **30 modelos Shallow ML**: 5 embeddings × 6 algoritmos
- **12 modelos Deep Learning**: Arquitecturas diversas
- **Total: 42 modelos diferentes**

### Cobertura Completa:
✅ Todos los embeddings: BoW, TF-IDF (word/char), Word2Vec, FastText, BERT  
✅ Todos los modelos: 6 Shallow ML + LSTM + GRU + CNN + Híbridos + BERT  
✅ Todas las técnicas de los PDFs aplicadas  
✅ Documentación completa con referencias

---

**¡PROYECTO COMPLETADO! 🎉**

Todas las combinaciones posibles de modelos × embeddings han sido implementadas siguiendo las técnicas de los PDFs y notebooks del curso.
