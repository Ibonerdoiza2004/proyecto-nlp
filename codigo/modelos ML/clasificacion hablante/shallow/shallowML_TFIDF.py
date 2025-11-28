import ast
import gc

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

# Configuración
np.random.seed(10)

print("SHALLOW ML + TF-IDF (PALABRAS)")

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

# Preparar datos
X = df["text_for_tfidf"].values
y = df["speaker"].values

# Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=10, stratify=y_encoded
)

# Vectorización con TF-IDF (palabras usando vectorizer existente)
vectorizer = joblib.load('models/vec_tfidf_word.joblib')

X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Definir clasificadores
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=10),
    'SVM Linear': LinearSVC(max_iter=2000, random_state=10),
    'SVM RBF': SVC(kernel='rbf', random_state=10),
    'Decision Tree': DecisionTreeClassifier(random_state=10),
    'Naive Bayes': MultinomialNB()
}

results = {}
trained_models = {}

for name, clf in classifiers.items():
    print(f"Modelo: {name}")
    
    # Entrenar
    clf.fit(X_train_tfidf, y_train)
    
    # Predecir
    y_pred = clf.predict(X_test_tfidf)
    
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
plt.title(f'Matriz de Confusión - {best_model_name} + TF-IDF (palabras)')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('confusion_matrix_shallow_tfidf.png', dpi=300, bbox_inches='tight')

# Gráfico de comparación de accuracy
plt.figure(figsize=(8, 6))
plt.barh(results_df['Modelo'], results_df['Accuracy'])
plt.xlabel('Accuracy')
plt.title('Comparación de Accuracy - Test Set')
plt.xlim([0, 1])
for i, v in enumerate(results_df['Accuracy']):
    plt.text(v + 0.01, i, f'{v:.4f}', va='center')
plt.tight_layout()
plt.savefig('comparison_shallowML_tfidf.png', dpi=300, bbox_inches='tight')
