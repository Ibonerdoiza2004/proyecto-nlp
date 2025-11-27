"""
Shallow ML con Word2Vec Trainable
Word2Vec con embeddings entrenables + 6 clasificadores tradicionales
"""

import ast, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*60)
print("SHALLOW ML + WORD2VEC TRAINABLE")
print("="*60)

df = pd.read_csv("dataset/dataset_preprocesado.csv")
df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(lambda x: x if isinstance(x, list) else ast.literal_eval(x) if x else [])
df = df[df["lemmas_no_stop"].apply(len) >= 3].copy()

texts, labels = df["lemmas_no_stop"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Entrenar Word2Vec trainable
print("\nEntrenando Word2Vec trainable...")
w2v_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=2, workers=4, sg=1, epochs=15)

def get_avg_word2vec(text, model):
    """Promedio de embeddings Word2Vec para clasificadores shallow"""
    vectors = [model.wv[word] for word in text if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

X_train_w2v = np.array([get_avg_word2vec(text, w2v_model) for text in X_train])
X_test_w2v = np.array([get_avg_word2vec(text, w2v_model) for text in X_test])

# Clasificadores
classifiers = {
    'Logistic Regression': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {'C': [0.1, 1, 10], 'penalty': ['l2']}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
    },
    'SVM Linear': {
        'model': LinearSVC(random_state=42, max_iter=2000),
        'params': {'C': [0.1, 1, 10]}
    },
    'SVM RBF': {
        'model': SVC(kernel='rbf', random_state=42),
        'params': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}
    },
    'Gaussian NB': {
        'model': GaussianNB(),
        'params': {'var_smoothing': [1e-9, 1e-8]}
    }
}

results = {}

for name, config in classifiers.items():
    print(f"\n{'='*60}")
    print(f"Entrenando: {name}")
    print(f"{'='*60}")
    
    grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_w2v, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_w2v)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'best_params': grid_search.best_params_,
        'predictions': y_pred
    }
    
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Matriz de confusión del mejor modelo
best_classifier = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n{'='*60}")
print(f"MEJOR MODELO: {best_classifier[0]} (Acc: {best_classifier[1]['accuracy']:.4f})")
print(f"{'='*60}")

cm = confusion_matrix(y_test, best_classifier[1]['predictions'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Shallow ML + Word2Vec Trainable - {best_classifier[0]}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_shallowML_word2vec_trainable.png', dpi=300)

# Comparación de todos los modelos
plt.figure(figsize=(12, 6))
models = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models]
plt.bar(models, accuracies, color='skyblue', edgecolor='navy')
plt.ylabel('Accuracy')
plt.title('Shallow ML + Word2Vec Trainable - Comparación')
plt.xticks(rotation=45, ha='right')
plt.ylim(min(accuracies) - 0.05, 1.0)
for i, (m, acc) in enumerate(zip(models, accuracies)):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('comparison_shallowML_word2vec_trainable.png', dpi=300)
print("\n✓ Completado")
