import ast
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB

# Configuración
np.random.seed(10)

print("SHALLOW ML + WORD2VEC")

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

# Filtrar frases cortas
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

# Preparar datos
texts = df["lemmas_no_stop"].tolist()
labels = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Split train/test
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)

w2v_model = Word2Vec.load('models/w2v.model')

# Función para convertir texto a vector promedio
def text_to_vector(text, model):
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
X_train = np.array([text_to_vector(text, w2v_model) for text in X_train_texts])
X_test = np.array([text_to_vector(text, w2v_model) for text in X_test_texts])

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definir clasificadores
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=10),
    'SVM Linear': LinearSVC(max_iter=2000, random_state=10),
    'SVM RBF': SVC(kernel='rbf', random_state=10),
    'Decision Tree': DecisionTreeClassifier(random_state=10),
    'Gaussian Naive Bayes': GaussianNB()
}

# Entrenar y evaluar cada clasificador
results = {}
trained_models = {}

for name, clf in classifiers.items():
    print(f"Modelo: {name}")
    
    # Entrenar
    clf.fit(X_train, y_train)
    
    # Predecir
    y_pred = clf.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    trained_models[name] = clf
    
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    # Liberar memoria
    del y_pred
    gc.collect()

results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()]
})

results_df = results_df.sort_values('Accuracy', ascending=False)

# Mejor modelo
best_model_name = results_df.iloc[0]['Modelo']
best_model = trained_models[best_model_name]
best_predictions = results[best_model_name]['predictions']

print(f"Mejor modelo: {best_model_name} (Acc: {results_df.iloc[0]['Accuracy']:.4f})")

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

# Gráfico de comparación de accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(results_df['Modelo'], results_df['Accuracy'])
plt.xlabel('Accuracy')
plt.title('Comparación de Accuracy - Test Set')
plt.xlim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')
plt.tight_layout()
plt.savefig('comparison_shallowML_w2v.png', dpi=300, bbox_inches='tight')