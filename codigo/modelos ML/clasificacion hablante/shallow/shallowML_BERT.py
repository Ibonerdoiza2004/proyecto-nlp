"""
Clasificación de Hablantes usando Shallow Machine Learning con BERT (mean pooling)
Compara 6 algoritmos: Logistic Regression, Random Forest, SVM Linear, SVM RBF, Decision Tree, Naive Bayes
Embedding: Promedio de embeddings BERT (BETO) para cada texto
Fuente: BERT como extractor de features + Shallow ML
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
from sklearn.naive_bayes import GaussianNB
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
import gc  # Para liberar memoria

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
np.random.seed(42)
torch.manual_seed(42)

BATCH_SIZE = 16  # Para procesar embeddings

print("="*60)
print("SHALLOW ML + BERT (BETO) MEAN POOLING")
print("="*60)

print("\nCargando datos...")
df = pd.read_csv("dataset/dataset_preprocesado.csv")

def parse_list(x):
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(parse_list)
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()
df["text"] = df["lemmas_no_stop"].apply(lambda x: " ".join(x))

print(f"Total de muestras: {len(df)}")
print(f"Distribución de hablantes:\n{df['speaker'].value_counts()}")

texts = df["text"].tolist()
labels = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

print(f"\nClases: {label_encoder.classes_}")
print(f"Número de clases: {num_classes}")

# Split train/test
pos_indices = np.arange(len(df))
train_pos, test_pos, y_train, y_test = train_test_split(pos_indices, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"\nTrain: {len(train_pos)} muestras")
print(f"Test: {len(test_pos)} muestras")

# Cargar embeddings ya calculados de BETO mean pooling
import os
bert_mean_path = os.path.join("models", "bert_mean.npz")
embeddings_npz = np.load(bert_mean_path)
all_embeddings = embeddings_npz[embeddings_npz.files[0]]

# Usar índices posicionales para embeddings
X_train = all_embeddings[train_pos]
X_test = all_embeddings[test_pos]

print(f"\nShape train: {X_train.shape}")
print(f"Shape test: {X_test.shape}")
print(f"Embedding dimension: {X_train.shape[1]}")

print(f"NaN en train: {np.isnan(X_train).sum()}")
print(f"NaN en test: {np.isnan(X_test).sum()}")

# Escalar los datos para mejor convergencia
print("\nEscalando datos...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir clasificadores (GaussianNB para datos continuos)
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM Linear': LinearSVC(max_iter=2000, random_state=42),
    'SVM RBF': SVC(kernel='rbf', random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gaussian Naive Bayes': GaussianNB()
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
plt.title(f'Matriz de Confusión - {best_model_name} + BERT')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_shallow_bert.png', dpi=300, bbox_inches='tight')
print("\nMatriz de confusión guardada en: confusion_matrix_shallow_bert.png")

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
plt.savefig('shallow_bert_comparison.png', dpi=300, bbox_inches='tight')
print("Gráfico de comparación guardado en: shallow_bert_comparison.png")

# Guardar modelos
print("\n" + "="*60)
print("GUARDANDO MODELOS")
print("="*60)

# Guardar embeddings (para no tener que recalcularlos)
np.save('models/bert_embeddings_train.npy', X_train)
np.save('models/bert_embeddings_test.npy', X_test)
print("BERT embeddings guardados en: models/bert_embeddings_*.npy")

joblib.dump(best_model, 'models/best_shallow_bert.joblib')
print(f"Mejor modelo ({best_model_name}) guardado en: models/best_shallow_bert.joblib")

joblib.dump(scaler, 'models/scaler_shallow_bert.joblib')
print("Scaler guardado en: models/scaler_shallow_bert.joblib")

# Guardar todos los resultados
joblib.dump(results, 'models/results_shallow_bert.joblib')
print("Resultados guardados en: models/results_shallow_bert.joblib")

# Función de predicción
def predecir_hablante_desde_embedding(embedding, classifier, label_encoder):
    """
    Predice el hablante a partir de un embedding BERT ya calculado (mean pooling).
    embedding: numpy array shape (1, D) o (D,)
    """
    emb = embedding.reshape(1, -1)
    prediccion = classifier.predict(emb)[0]
    proba = classifier.predict_proba(emb)[0] if hasattr(classifier, 'predict_proba') else None
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


print("\n(Ejemplo de predicción omitido: ahora se usan embeddings precalculados)")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)
print("\nRESUMEN:")
print(f"- Embedding: BERT mean pooling ({X_train.shape[1]}D)")
print(f"- Embeddings used: models/bert_mean.npz")
print(f"- Modelos entrenados: {len(classifiers)}")
print(f"- Mejor modelo: {best_model_name}")
print(f"- Mejor accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"- Tiempo total: {sum(r['train_time'] for r in results.values()):.2f}s")
print("\nVENTAJA DE BERT:")
print("- Embeddings contextuales (considera el contexto completo)")
print("- Pre-entrenado en gran corpus de español")
print("- Captura relaciones semánticas profundas")
print("- Estado del arte en representación de texto")
