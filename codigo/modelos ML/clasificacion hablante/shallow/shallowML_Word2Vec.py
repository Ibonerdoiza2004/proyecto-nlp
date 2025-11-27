"""
Clasificación de Hablantes usando Shallow Machine Learning con Word2Vec
Compara 6 algoritmos: Logistic Regression, Random Forest, SVM Linear, SVM RBF, Decision Tree, Naive Bayes
Embedding: Promedio de vectores Word2Vec para cada texto
Fuente: Práctica "Classification_using_shallow_machine_learning_techniques.ipynb"
"""

import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB  # GaussianNB para datos continuos
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
import gc

# Configuración
np.random.seed(42)

print("="*60)
print("SHALLOW ML + WORD2VEC")
print("="*60)

print("\nCargando datos...")
# Cargar dataset preprocesado
df = pd.read_csv("dataset/dataset_preprocesado.csv")

# Parsear lemmas
def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)

# Filtrar frases muy cortas (menos de 3 palabras)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

print(f"Total de muestras: {len(df)}")
print(f"Distribución de hablantes:\n{df['speaker'].value_counts()}")

# Preparar datos
texts = df["lemmas_no_stop"].tolist()
labels = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"\nClases: {label_encoder.classes_}")
print(f"Número de clases: {num_classes}")

# Split train/test
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {len(X_train_texts)} muestras")
print(f"Test: {len(X_test_texts)} muestras")

# Cargar Word2Vec pre-entrenado
print("\n" + "="*60)
print("CARGANDO WORD2VEC PRE-ENTRENADO")
print("="*60)

w2v_model = Word2Vec.load('models/w2v.model')

vocab_size = len(w2v_model.wv)
print(f"Vocabulario: {vocab_size} palabras")

# Función para convertir texto a vector promedio
def text_to_vector(text, model):
    """
    Convierte un texto (lista de palabras) en un vector promediando los embeddings Word2Vec
    """
    vectors = []
    for word in text:
        if word in model.wv:
            vectors.append(model.wv[word])
    
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        # Si no hay palabras en vocabulario, retornar vector cero
        return np.zeros(model.vector_size)

# Convertir textos a vectores
print("\n" + "="*60)
print("VECTORIZACIÓN: PROMEDIO WORD2VEC")
print("="*60)

print("\nConvirtiendo textos a vectores...")
X_train = np.array([text_to_vector(text, w2v_model) for text in X_train_texts])
X_test = np.array([text_to_vector(text, w2v_model) for text in X_test_texts])

print(f"Shape train: {X_train.shape}")
print(f"Shape test: {X_test.shape}")
print(f"Dimensión de embeddings: {w2v_model.vector_size}")

# Verificar que no haya NaN
print(f"NaN en train: {np.isnan(X_train).sum()}")
print(f"NaN en test: {np.isnan(X_test).sum()}")

# Escalar los datos para mejor convergencia
print("\nEscalando datos...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir clasificadores (GaussianNB para datos continuos)
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM Linear': LinearSVC(max_iter=2000, random_state=42),
    'SVM RBF': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gaussian Naive Bayes': GaussianNB()  # Para datos continuos
}

# Entrenar y evaluar cada clasificador
print("\n" + "="*60)
print("ENTRENAMIENTO Y EVALUACIÓN")
print("="*60)

results = {}
trained_models = {}

for name, clf in classifiers.items():
    print(f"\n{'='*60}")
    print(f"Modelo: {name}")
    print(f"{'='*60}")
    
    # Entrenar
    start_time = time()
    clf.fit(X_train, y_train)
    train_time = time() - start_time
    
    # Predecir
    start_time = time()
    y_pred = clf.predict(X_test)
    test_time = time() - start_time
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'train_time': train_time,
        'test_time': test_time,
        'predictions': y_pred
    }
    
    trained_models[name] = clf
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tiempo entrenamiento: {train_time:.2f}s")
    print(f"Tiempo predicción: {test_time:.4f}s")
    
    # Liberar memoria
    del y_pred
    gc.collect()

# Comparación de resultados
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS")
print("="*60)

results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Train Time (s)': [r['train_time'] for r in results.values()],
    'Test Time (s)': [r['test_time'] for r in results.values()]
})

results_df = results_df.sort_values('Accuracy', ascending=False)
print("\n", results_df.to_string(index=False))

# Mejor modelo
best_model_name = results_df.iloc[0]['Modelo']
best_model = trained_models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"\n🏆 Mejor modelo: {best_model_name}")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

# Reporte detallado del mejor modelo
print("\n" + "="*60)
print(f"REPORTE DETALLADO - {best_model_name}")
print("="*60)
print(classification_report(
    y_test, best_predictions,
    target_names=label_encoder.classes_
))

# Matriz de confusión del mejor modelo
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title(f'Matriz de Confusión - {best_model_name} + Word2Vec')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_shallow_w2v.png', dpi=300, bbox_inches='tight')
print("\nMatriz de confusión guardada en: confusion_matrix_shallow_w2v.png")

# Gráfico de comparación de accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(results_df['Modelo'], results_df['Accuracy'])
plt.xlabel('Accuracy')
plt.title('Comparación de Accuracy - Test Set')
plt.xlim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')

plt.subplot(1, 2, 2)
plt.barh(results_df['Modelo'], results_df['Train Time (s)'])
plt.xlabel('Tiempo (segundos)')
plt.title('Tiempo de Entrenamiento')
for i, v in enumerate(results_df['Train Time (s)']):
    plt.text(v + 0.01, i, f'{v:.2f}s', va='center', fontsize=8)

plt.tight_layout()
plt.savefig('shallow_w2v_comparison.png', dpi=300, bbox_inches='tight')
print("Gráfico de comparación guardado en: shallow_w2v_comparison.png")

# Guardar modelos
print("\n" + "="*60)
print("GUARDANDO MODELOS")
print("="*60)

joblib.dump(best_model, 'models/best_shallow_w2v.joblib')
print(f"Mejor modelo ({best_model_name}) guardado en: models/best_shallow_w2v.joblib")

joblib.dump(label_encoder, 'models/label_encoder_shallow.joblib')
print("Label encoder guardado en: models/label_encoder_shallow.joblib")

# Guardar todos los resultados
joblib.dump(results, 'models/results_shallow_w2v.joblib')
print("Resultados guardados en: models/results_shallow_w2v.joblib")

# Función de predicción
def predecir_hablante_w2v(texto_lemmatizado, w2v_model, classifier, label_encoder):
    """
    Predice el hablante usando Word2Vec + Shallow ML
    texto_lemmatizado: lista de palabras ya lemmatizadas
    """
    # Convertir a vector
    vector = text_to_vector(texto_lemmatizado, w2v_model)
    vector = vector.reshape(1, -1)
    
    # Predecir
    prediccion = classifier.predict(vector)[0]
    proba = classifier.predict_proba(vector)[0] if hasattr(classifier, 'predict_proba') else None
    
    hablante = label_encoder.inverse_transform([prediccion])[0]
    confianza = proba[prediccion] if proba is not None else None
    
    return hablante, confianza

# Ejemplos de predicción
print("\n" + "="*60)
print("EJEMPLOS DE PREDICCIÓN")
print("="*60)

ejemplos = [
    ["hoy", "vamos", "analizar", "decisiones", "entrenadores"],
    ["estar", "acuerdo", "eso"],
    ["real", "madrid", "mejorar"],
]

for ejemplo in ejemplos:
    hablante, confianza = predecir_hablante_w2v(
        ejemplo, w2v_model, best_model, label_encoder
    )
    if confianza:
        print(f"\nTexto: {' '.join(ejemplo)}")
        print(f"Predicción: {hablante} (confianza: {confianza:.2%})")
    else:
        print(f"\nTexto: {' '.join(ejemplo)}")
        print(f"Predicción: {hablante}")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
print("\nRESUMEN:")
print(f"- Embedding: Word2Vec promedio ({w2v_model.vector_size}D)")
print(f"- Vocabulario: {vocab_size} palabras")
print(f"- Modelos entrenados: {len(classifiers)}")
print(f"- Mejor modelo: {best_model_name}")
print(f"- Mejor accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"- Tiempo total: {sum(r['train_time'] for r in results.values()):.2f}s")
print("\nVENTAJA DE WORD2VEC:")
print("- Captura similitud semántica entre palabras")
print("- Vectores densos (vs sparse en BoW/TF-IDF)")
print("- Reduce dimensionalidad manteniendo información")
