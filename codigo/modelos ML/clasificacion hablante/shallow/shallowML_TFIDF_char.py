"""
Clasificación de Hablantes usando Shallow Machine Learning con TF-IDF (caracteres n-grams)
Compara 6 algoritmos: Logistic Regression, Random Forest, SVM Linear, SVM RBF, Decision Tree, Naive Bayes
Fuente: Práctica "Classification_using_shallow_machine_learning_techniques.ipynb" + PDF págs 15-17
"""

import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
import gc

# Configuración
np.random.seed(42)

print("="*60)
print("SHALLOW ML + TF-IDF (CARACTERES N-GRAMS)")
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

# Convertir lemmas a texto para vectorización
df["text_for_tfidf"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))

print(f"Total de muestras: {len(df)}")
print(f"Distribución de hablantes:\n{df['speaker'].value_counts()}")

# Preparar datos
X = df["text_for_tfidf"].values
y = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

print(f"\nClases: {label_encoder.classes_}")
print(f"Número de clases: {num_classes}")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {len(X_train)} muestras")
print(f"Test: {len(X_test)} muestras")

# Vectorización con TF-IDF (caracteres n-grams usando vectorizer existente)
print("\n" + "="*60)
print("VECTORIZACIÓN: TF-IDF (CARACTERES N-GRAMS)")
print("="*60)

print("\nCargando vectorizer existente...")
vectorizer = joblib.load('models/vec_tfidf_char.joblib')

print("\nTransformando datos...")
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Vocabulario: {len(vectorizer.vocabulary_)} n-gramas de caracteres")
print(f"Shape train: {X_train_tfidf.shape}")
print(f"Shape test: {X_test_tfidf.shape}")
print(f"Sparsity: {(1.0 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])) * 100:.2f}%")

# Top n-gramas por TF-IDF
feature_names = vectorizer.get_feature_names_out()
print(f"\nEjemplo de features: {list(feature_names[:15])}")

# Definir clasificadores (según práctica de shallow ML)
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM Linear': LinearSVC(max_iter=2000, random_state=42),
    'SVM RBF': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': MultinomialNB()
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
    clf.fit(X_train_tfidf, y_train)
    train_time = time() - start_time
    
    # Predecir
    start_time = time()
    y_pred = clf.predict(X_test_tfidf)
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
plt.title(f'Matriz de Confusión - {best_model_name} + TF-IDF (char n-grams)')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_shallow_tfidf_char.png', dpi=300, bbox_inches='tight')
print("\nMatriz de confusión guardada en: confusion_matrix_shallow_tfidf_char.png")

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
plt.savefig('shallow_tfidf_char_comparison.png', dpi=300, bbox_inches='tight')
print("Gráfico de comparación guardado en: shallow_tfidf_char_comparison.png")

# Análisis de importancia de features (para modelos que lo soporten)
if hasattr(best_model, 'coef_'):
    print("\n" + "="*60)
    print("TOP N-GRAMAS DE CARACTERES POR CLASE")
    print("="*60)
    
    feature_names = vectorizer.get_feature_names_out()
    
    for idx, clase in enumerate(label_encoder.classes_):
        if len(best_model.coef_.shape) > 1:
            coef = best_model.coef_[idx]
        else:
            coef = best_model.coef_
        
        top_indices = np.argsort(coef)[-10:][::-1]
        top_features = [feature_names[i] for i in top_indices]
        top_values = [coef[i] for i in top_indices]
        
        print(f"\nClase: {clase}")
        for feature, value in zip(top_features, top_values):
            print(f"  '{feature}': {value:8.4f}")

# Guardar mejor modelo
print("\n" + "="*60)
print("GUARDANDO MODELOS")
print("="*60)

joblib.dump(best_model, 'models/best_shallow_tfidf_char.joblib')
print(f"Mejor modelo ({best_model_name}) guardado en: models/best_shallow_tfidf_char.joblib")

joblib.dump(label_encoder, 'models/label_encoder_shallow.joblib')
print("Label encoder guardado en: models/label_encoder_shallow.joblib")

# Guardar todos los resultados
joblib.dump(results, 'models/results_shallow_tfidf_char.joblib')
print("Resultados guardados en: models/results_shallow_tfidf_char.joblib")

# Función de predicción
def predecir_hablante_tfidf_char(texto, vectorizer, model, label_encoder):
    """
    Predice el hablante usando TF-IDF de caracteres
    """
    # Preprocesar (asumir que ya está lemmatizado)
    texto_vectorizado = vectorizer.transform([texto])
    prediccion = model.predict(texto_vectorizado)[0]
    proba = model.predict_proba(texto_vectorizado)[0] if hasattr(model, 'predict_proba') else None
    
    hablante = label_encoder.inverse_transform([prediccion])[0]
    confianza = proba[prediccion] if proba is not None else None
    
    return hablante, confianza

# Ejemplos de predicción
print("\n" + "="*60)
print("EJEMPLOS DE PREDICCIÓN")
print("="*60)

ejemplos = [
    "hoy vamos analizar decisiones entrenadores",
    "estar acuerdo eso",
    "real madrid mejorar",
]

for ejemplo in ejemplos:
    hablante, confianza = predecir_hablante_tfidf_char(
        ejemplo, vectorizer, best_model, label_encoder
    )
    if confianza:
        print(f"\nTexto: '{ejemplo}'")
        print(f"Predicción: {hablante} (confianza: {confianza:.2%})")
    else:
        print(f"\nTexto: '{ejemplo}'")
        print(f"Predicción: {hablante}")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
print("\nRESUMEN:")
print(f"- Embedding: TF-IDF caracteres ({len(vectorizer.vocabulary_)} features)")
print(f"- N-gramas: caracteres {vectorizer.ngram_range}")
print(f"- Modelos entrenados: {len(classifiers)}")
print(f"- Mejor modelo: {best_model_name}")
print(f"- Mejor accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"- Tiempo total: {sum(r['train_time'] for r in results.values()):.2f}s")
print("\nVENTAJA DE CHAR N-GRAMS:")
print("- Captura patrones a nivel de caracteres")
print("- Robusto ante errores ortográficos")
print("- Útil para detectar estilos de escritura")
