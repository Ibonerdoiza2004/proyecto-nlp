import ast
import gc
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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

# Filtrar frases muy cortas
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

# Vectorización con TF-IDF
try:
    vectorizer = joblib.load('models/vec_tfidf_word.joblib')
    X_train_tfidf = vectorizer.transform(X_train)
except:
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

# Definir clasificadores
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=10),
    'SVM Linear': LinearSVC(max_iter=2000, random_state=10, dual='auto'),
    'SVM RBF': SVC(kernel='rbf', random_state=10),
    'Decision Tree': DecisionTreeClassifier(random_state=10),
    'Naive Bayes': MultinomialNB()
}

results = {}
trained_models = {}

print("\nENTRENAMIENTO")

for name, clf in classifiers.items():
    print(f"Modelo: {name}")
    
    # Entrenar
    clf.fit(X_train_tfidf, y_train)
    
    # Predecir
    y_pred = clf.predict(X_test_tfidf)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results[name] = {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': y_pred
    }
    
    trained_models[name] = clf
    
    print(f"  Accuracy: {accuracy:.4f} | F1-Macro: {f1:.4f}")
    
    # Liberar memoria
    del y_pred
    gc.collect()


# DataFrame de resultados
results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'F1-Score': [r['f1'] for r in results.values()]
})

# Ordenar por F1-Score
results_df = results_df.sort_values('F1-Score', ascending=False)

print("\nRESULTADOS FINALES")
print(results_df)

# Mejor modelo
best_model_name = results_df.iloc[0]['Modelo']
best_model = trained_models[best_model_name]
best_predictions = results[best_model_name]['predictions']
best_f1 = results_df.iloc[0]['F1-Score']

print(f"\nMejor modelo seleccionado: {best_model_name} (F1: {best_f1:.4f})")
print(classification_report(y_test, best_predictions, target_names=label_encoder.classes_, zero_division=0))

# Matriz de confusión del mejor modelo
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title(f'Matriz de Confusión - {best_model_name} + TF-IDF Word (F1: {best_f1:.2f})')
plt.ylabel('Real')
plt.xlabel('Predicción')
plt.tight_layout()
plt.savefig('imagenes/confusion_matrix_shallow_tfidf.png', dpi=300, bbox_inches='tight')

# Gráfico de comparación de F1-Score
plt.figure(figsize=(10, 6))
bars = plt.barh(results_df['Modelo'], results_df['F1-Score'], color='teal')
plt.xlabel('F1-Score (Macro)')
plt.title('Comparación de Modelos ML Clásicos + TF-IDF (Palabras)')
plt.xlim([0, 1])

for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{width:.4f}', va='center')

plt.tight_layout()
plt.savefig('imagenes/comparison_shallowML_tfidf.png', dpi=300, bbox_inches='tight')

# Guardar el mejor modelo
joblib.dump(best_model, 'models/clasificacion_hablantes/best_shallow_tfidf.joblib')