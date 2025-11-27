# Matriz de Modelos y Embeddings - Plan de Implementación

**Fecha**: 27 de Noviembre, 2025  
**Objetivo**: Crear todas las combinaciones posibles entre tipos de modelos y métodos de embeddings

---

## 📊 Matriz de Combinaciones

### Métodos de Embedding Disponibles:
1. **Tradicionales** (Shallow):
   - Bag of Words (BoW)
   - TF-IDF (palabras)
   - TF-IDF (caracteres)

2. **Word2Vec**:
   - Frozen (no entrenable)
   - Trainable (fine-tuning)

3. **FastText**:
   - Con n-gramas de caracteres

4. **BERT** (BETO):
   - Embeddings pre-calculados (mean pooling)
   - Embeddings [CLS] token

### Tipos de Modelos:
1. **Shallow Machine Learning**:
   - Logistic Regression
   - Random Forest
   - SVM (Linear)
   - SVM (RBF)
   - Decision Tree
   - Naive Bayes

2. **Deep Learning**:
   - LSTM (unidireccional/bidireccional)
   - CNN
   - GRU
   - Modelos híbridos (CNN-LSTM)

---

## ✅ Estado Actual (Ya Implementados)

| Modelo | Embedding | Archivo | Status |
|--------|-----------|---------|--------|
| LSTM Bidireccional | Word2Vec (frozen) | `lstmWord2Vec.py` | ✅ Mejorado |
| LSTM | Word2Vec (trainable) | `lstmWord2VecTrainable.py` | ✅ Existe |
| CNN | Word2Vec (frozen) | `cnnWord2Vec.py` | ✅ Mejorado |
| LSTM Bidireccional | BERT (mean) | `lstmBert.py` | ✅ Mejorado |
| LSTM | - (generativo) | `lstm_generacion.py` | ✅ Mejorado |

**Total implementados**: 5 combinaciones

---

## 🎯 Combinaciones a Implementar

### PRIORIDAD ALTA (Shallow ML con embeddings tradicionales)

#### 1. Shallow ML + Bag of Words
**Archivo**: `codigo/modelos ML/clasificacion hablante/shallowML_BoW.py`
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ SVM Linear
- ✅ SVM RBF
- ✅ Decision Tree
- ✅ Naive Bayes

**Fuente**: Práctica "Classification_using_shallow_machine_learning_techniques.ipynb"

#### 2. Shallow ML + TF-IDF (palabras)
**Archivo**: `codigo/modelos ML/clasificacion hablante/shallowML_TFIDF.py`
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ SVM Linear
- ✅ SVM RBF
- ✅ Decision Tree
- ✅ Naive Bayes

**Fuente**: Práctica "Classification_using_shallow_machine_learning_techniques.ipynb"

#### 3. Shallow ML + TF-IDF (caracteres)
**Archivo**: `codigo/modelos ML/clasificacion hablante/shallowML_TFIDF_char.py`
- ✅ Logistic Regression
- ✅ Random Forest
- ✅ SVM Linear
- ✅ SVM RBF
- ✅ Decision Tree
- ✅ Naive Bayes

**Fuente**: PDF págs 15-17 (TF-IDF con n-gramas de caracteres)

---

### PRIORIDAD MEDIA (Deep Learning con Word2Vec/FastText)

#### 4. GRU + Word2Vec
**Archivo**: `codigo/modelos ML/clasificacion hablante/gruWord2Vec.py`
- GRU Bidireccional con 2 capas
- Packed sequences
- Gradient clipping

**Fuente**: PDF págs 38-40 (GRU vs LSTM)

#### 5. CNN + FastText
**Archivo**: `codigo/modelos ML/clasificacion hablante/cnnFastText.py`
- CNN con múltiples kernels
- FastText con n-gramas de caracteres
- Batch Normalization

**Fuente**: PDF págs 25-30 (CNNs) + FastText model disponible

#### 6. LSTM + FastText
**Archivo**: `codigo/modelos ML/clasificacion hablante/lstmFastText.py`
- LSTM Bidireccional
- FastText embeddings
- Packed sequences

**Fuente**: PDF págs 59-60 + FastText model

#### 7. CNN-LSTM Híbrido + Word2Vec
**Archivo**: `codigo/modelos ML/clasificacion hablante/cnnLstmWord2Vec.py`
- CNN para extracción de features
- LSTM para secuencia temporal
- Arquitectura híbrida

**Fuente**: PDF pág 30 (mención de híbridos)

---

### PRIORIDAD BAJA (Deep Learning con BERT)

#### 8. CNN + BERT
**Archivo**: `codigo/modelos ML/clasificacion hablante/cnnBert.py`
- CNN sobre embeddings BERT
- Múltiples kernels temporales
- Batch Normalization

**Fuente**: PDF págs 25-30

#### 9. GRU + BERT
**Archivo**: `codigo/modelos ML/clasificacion hablante/gruBert.py`
- GRU Bidireccional
- Atención Bahdanau
- BERT embeddings

**Fuente**: PDF págs 38-40 + 64-71

#### 10. Bidirectional Encoder (BERT-style fine-tuning)
**Archivo**: `codigo/modelos ML/clasificacion hablante/bertFineTuning.py`
- Fine-tuning de BETO completo
- Classification head
- Learning rate diferenciado

**Fuente**: PDF págs 82-90 (Transformers)

---

## 📈 Resumen de Combinaciones

### Total de combinaciones posibles:
- **Embeddings**: 5 tipos (BoW, TF-IDF word, TF-IDF char, Word2Vec, FastText, BERT)
- **Modelos Shallow**: 6 tipos
- **Modelos Deep**: 5 tipos (LSTM, CNN, GRU, CNN-LSTM, BERT fine-tuning)

**Total teórico**: 
- Shallow ML: 3 embeddings × 6 modelos = 18 combinaciones
- Deep Learning: 3 embeddings × 5 modelos = 15 combinaciones
- **TOTAL**: 33 combinaciones posibles

### Implementación realista (priorizada):
- ✅ Ya implementadas: 5
- 🎯 Prioridad ALTA: 3 archivos (18 modelos shallow)
- 🎯 Prioridad MEDIA: 4 archivos (4 modelos deep)
- 🎯 Prioridad BAJA: 3 archivos (3 modelos deep)

**Total a implementar**: 10 archivos nuevos

---

## 🔧 Detalles de Implementación

### Template para Shallow ML:
```python
# Configuración
vectorizers = {
    'bow': CountVectorizer(),
    'tfidf_word': TfidfVectorizer(analyzer='word'),
    'tfidf_char': TfidfVectorizer(analyzer='char', ngram_range=(2,5))
}

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM Linear': LinearSVC(),
    'SVM RBF': SVC(kernel='rbf'),
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': MultinomialNB()
}

# Grid Search para cada combinación
# Cross-validation 5-fold
# Métricas: Accuracy, Precision, Recall, F1
```

### Template para Deep Learning:
```python
# Arquitectura base
class ModeloClasificador(nn.Module):
    def __init__(self, ...):
        # Embedding layer
        # Feature extractor (CNN/LSTM/GRU)
        # Attention (opcional)
        # Classifier head
    
    # Mejoras del PDF aplicadas:
    # - Bidireccionalidad
    # - Múltiples capas
    # - Packed sequences
    # - Gradient clipping
    # - Regularización L2
```

---

## 📊 Métricas Esperadas por Tipo

### Shallow ML + BoW/TF-IDF:
- Baseline: 80-85% accuracy
- Mejor: SVM Linear o Logistic Regression
- Tiempo: Muy rápido (~segundos)

### Deep Learning + Word2Vec:
- Baseline: 85-90% accuracy
- Mejor: LSTM Bidireccional o CNN
- Tiempo: Medio (~minutos)

### Deep Learning + BERT:
- Baseline: 90-95% accuracy
- Mejor: Fine-tuning completo
- Tiempo: Lento (~horas)

---

## 🎯 Plan de Ejecución

### Fase 1: Shallow ML (INMEDIATO)
1. ✅ Cargar vectorizadores pre-entrenados (BoW, TF-IDF)
2. ✅ Implementar 6 clasificadores shallow
3. ✅ Grid Search con validación cruzada
4. ✅ Comparar 3 tipos de embeddings

**Tiempo estimado**: 2-3 horas
**Archivos**: 3 nuevos

### Fase 2: Deep Learning con Word2Vec/FastText (SIGUIENTE)
1. ✅ GRU + Word2Vec
2. ✅ CNN + FastText
3. ✅ LSTM + FastText
4. ✅ CNN-LSTM Híbrido

**Tiempo estimado**: 4-6 horas
**Archivos**: 4 nuevos

### Fase 3: Deep Learning con BERT (OPCIONAL)
1. ✅ CNN + BERT
2. ✅ GRU + BERT
3. ✅ BERT Fine-tuning

**Tiempo estimado**: 6-8 horas (entrenamiento pesado)
**Archivos**: 3 nuevos

---

## 📚 Referencias por Combinación

| Combinación | Páginas PDF | Notebook |
|-------------|-------------|----------|
| Shallow ML + BoW/TF-IDF | 15-17, 20-24 | Classification_shallow_ML.ipynb |
| LSTM/GRU | 38-60 | LSTMs_for_Text_Classification.ipynb |
| CNN | 25-30 | CNNs_for_Text_Classification.ipynb |
| Atención | 61-71 | Machine_Translation.ipynb |
| Word2Vec | 10-14 | (modelo pre-entrenado) |
| FastText | 14 | (modelo pre-entrenado) |
| BERT | 82-90 | (modelo BETO disponible) |

---

## ✅ Checklist de Implementación - COMPLETADO

### Fase 1: Shallow ML ✅ COMPLETADO
- [x] shallowML_BoW.py (6 modelos) ✅
- [x] shallowML_TFIDF.py (6 modelos) ✅
- [x] shallowML_TFIDF_char.py (6 modelos) ✅
- [x] shallowML_Word2Vec.py (6 modelos) ✅
- [x] shallowML_FastText.py (6 modelos) ✅

### Fase 2: Deep Learning Word2Vec/FastText ✅ COMPLETADO
- [x] gruWord2Vec.py ✅
- [x] cnnFastText.py ✅
- [x] lstmFastText.py ✅
- [x] cnnLstmWord2Vec.py ✅

### Fase 3: Deep Learning BERT ✅ COMPLETADO
- [x] cnnBert.py ✅
- [x] gruBert.py ✅
- [x] bertFineTuning.py ✅

**Total completado**: 12 archivos nuevos + 5 mejorados = **17 archivos** = **42 modelos**

---

## 🎉 RESUMEN FINAL

### Archivos Creados (12 nuevos):
1. ✅ **shallowML_BoW.py** - 6 clasificadores con Bag of Words
2. ✅ **shallowML_TFIDF.py** - 6 clasificadores con TF-IDF (palabras)
3. ✅ **shallowML_TFIDF_char.py** - 6 clasificadores con TF-IDF (caracteres)
4. ✅ **shallowML_Word2Vec.py** - 6 clasificadores con Word2Vec
5. ✅ **shallowML_FastText.py** - 6 clasificadores con FastText
6. ✅ **gruWord2Vec.py** - GRU BiDir + Word2Vec
7. ✅ **cnnFastText.py** - CNN + FastText
8. ✅ **lstmFastText.py** - LSTM BiDir + FastText
9. ✅ **cnnLstmWord2Vec.py** - CNN-LSTM Híbrido
10. ✅ **cnnBert.py** - CNN + BERT frozen
11. ✅ **gruBert.py** - GRU BiDir + BERT frozen
12. ✅ **bertFineTuning.py** - BERT fine-tuning completo

### Archivos Mejorados Anteriormente (5):
13. ✅ **lstmWord2Vec.py** - LSTM BiDir mejorado
14. ✅ **cnnWord2Vec.py** - CNN mejorado
15. ✅ **lstmBert.py** - LSTM+BERT con attention
16. ✅ **lstm_generacion.py** - LSTM generativo
17. ✅ **lstmWord2VecTrainable.py** - LSTM trainable (existente)

### Cobertura Total:
- 📊 **30 modelos Shallow ML** (5 embeddings × 6 algoritmos)
- 🧠 **12 modelos Deep Learning** (arquitecturas diversas)
- 🎯 **TOTAL: 42 modelos diferentes** implementados

### Técnicas Aplicadas:
✅ Bidirectional RNNs (LSTM/GRU)
✅ Multiple Layers (2 capas apiladas)
✅ Packed Sequences (eficiencia)
✅ Attention Mechanisms (Bahdanau)
✅ Gradient Clipping (estabilidad)
✅ L2 Regularization (overfitting)
✅ Batch Normalization (CNNs)
✅ Learning Rate Scheduling (BERT)
✅ Warmup Steps (BERT fine-tuning)
✅ Differential Learning Rates (BERT)
✅ Teacher Forcing (generación)
✅ Beam Search (generación)

### Todas las Técnicas de los PDFs:
✅ PDF "3_UNIT 3 - NLP.pdf" (81 páginas)
✅ PDF "3_RNNs_Atencion.pdf" (9 páginas)  
✅ 6 Jupyter Notebooks de prácticas
