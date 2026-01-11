import gc
import os
import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(10)
np.random.seed(10)

print("SHALLOW ML + BETO CLS")

df = pd.read_csv("dataset/dataset_bert.csv")

import ast
def parse_embedding(x):
    try:
        if isinstance(x, str):
            return ast.literal_eval(x)
        return x
    except:
        return []

df['bert_cls'] = df['bert_cls'].apply(parse_embedding)

df = df[df["text"].str.len() >= 5].copy()

texts, labels = df["text"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

all_embeddings = np.array(df['bert_cls'].tolist())

X_train_bert_cls, X_test_bert_cls, y_train, y_test = train_test_split(
    all_embeddings, labels_encoded, test_size=0.2, random_state=10, stratify=labels_encoded
)

# Clasificadores
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
    print(name)
    
    model.fit(X_train_bert_cls, y_train)
    y_pred = model.predict(X_test_bert_cls)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    results[name] = {
        'accuracy': accuracy,
        'f1': f1,
        'predictions': y_pred,
        'model': model
    }
    
    print(f"Accuracy: {accuracy:.4f} | F1 Macro: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))
    
    del y_pred
    gc.collect()

# Matriz de confusión del mejor modelo
best_classifier_name, best_classifier_data = max(results.items(), key=lambda x: x[1]['f1'])
print(f"Mejor modelo: {best_classifier_name} (F1: {best_classifier_data['f1']:.4f})")

# Guardar el mejor modelo
joblib.dump(best_classifier_data['model'], 'models/clasificacion_hablantes/best_shallow_bert_cls.joblib')

cm = confusion_matrix(y_test, best_classifier_data['predictions'])
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Shallow ML + BETO CLS - {best_classifier_name} (F1: {best_classifier_data["f1"]:.2f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('imagenes/confusion_matrix_shallow_bert_cls.png', dpi=300)

# Comparación de todos los modelos
plt.figure(figsize=(12, 6))
models = list(results.keys())
f1_scores = [results[m]['f1'] for m in models]
plt.bar(models, f1_scores, color='mediumpurple', edgecolor='indigo')
plt.ylabel('F1 Score (Macro)')
plt.title('Shallow ML + BETO CLS - Comparación (F1)')
plt.xticks(rotation=45, ha='right')
plt.ylim(min(f1_scores) - 0.05, 1.0)
for i, (m, score) in enumerate(zip(models, f1_scores)):
    plt.text(i, score + 0.01, f'{score:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('imagenes/comparison_shallowML_bert_cls.png', dpi=300)
