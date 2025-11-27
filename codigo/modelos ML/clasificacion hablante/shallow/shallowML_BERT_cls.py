"""
Shallow ML con BERT CLS Token
Usa solo el embedding del token [CLS] de BERT + 6 clasificadores tradicionales
"""

import numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

print("="*60)
print("SHALLOW ML + BERT CLS TOKEN")
print("="*60)

df = pd.read_csv("dataset/dataset_bert.csv")
df = df[df["text"].str.len() >= 10].copy()

texts, labels = df["text"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Cargar embeddings ya calculados de BETO CLS
import os
bert_cls_path = os.path.join("models", "bert_cls.npz")
embeddings_npz = np.load(bert_cls_path)
all_embeddings = embeddings_npz[embeddings_npz.files[0]]

# Split using positional indices to avoid index-alignment issues
pos_indices = np.arange(len(df))
train_pos, test_pos, y_train, y_test = train_test_split(pos_indices, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Get embeddings using positional indices
X_train_bert_cls = all_embeddings[train_pos]
X_test_bert_cls = all_embeddings[test_pos]

print(f"BERT CLS embedding dim: {X_train_bert_cls.shape[1]}")

# Clasificadores con parámetros fijos (sin GridSearch para ahorrar memoria)
classifiers = {
    'Logistic Regression': LogisticRegression(C=1, penalty='l2', max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'SVM Linear': LinearSVC(C=1, max_iter=2000, random_state=42),
    'SVM RBF': SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=2, random_state=42),
    'Gaussian NB': GaussianNB(var_smoothing=1e-9)
}

results = {}

for name, model in classifiers.items():
    print(f"\n{'='*60}")
    print(f"Entrenando: {name}")
    print(f"{'='*60}")
    
    model.fit(X_train_bert_cls, y_train)
    y_pred = model.predict(X_test_bert_cls)
    
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }
    
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Liberar memoria
    del y_pred
    gc.collect()

# Matriz de confusión del mejor modelo
best_classifier = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n{'='*60}")
print(f"MEJOR MODELO: {best_classifier[0]} (Acc: {best_classifier[1]['accuracy']:.4f})")
print(f"{'='*60}")

cm = confusion_matrix(y_test, best_classifier[1]['predictions'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Shallow ML + BERT CLS Token - {best_classifier[0]}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_shallowML_bert_cls.png', dpi=300)

# Comparación de todos los modelos
plt.figure(figsize=(12, 6))
models = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models]
plt.bar(models, accuracies, color='mediumpurple', edgecolor='indigo')
plt.ylabel('Accuracy')
plt.title('Shallow ML + BERT CLS Token - Comparación')
plt.xticks(rotation=45, ha='right')
plt.ylim(min(accuracies) - 0.05, 1.0)
for i, (m, acc) in enumerate(zip(models, accuracies)):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('comparison_shallowML_bert_cls.png', dpi=300)
print("\n✓ Completado")
