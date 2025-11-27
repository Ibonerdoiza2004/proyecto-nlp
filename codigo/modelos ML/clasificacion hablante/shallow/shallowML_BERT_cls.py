"""
Shallow ML con BERT CLS Token
Usa solo el embedding del token [CLS] de BERT + 6 clasificadores tradicionales
"""

import ast, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

BERT_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LEN = 128

print("="*60)
print("SHALLOW ML + BERT CLS TOKEN")
print("="*60)

df = pd.read_csv("dataset/dataset_bert.csv")
df = df[df["text"].str.len() >= 10].copy()

texts, labels = df["text"].tolist(), df["speaker"].values
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# Cargar BERT
print("Cargando BERT...")
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModel.from_pretrained(BERT_MODEL).to(device)

# Congelar BERT
for param in bert_model.parameters():
    param.requires_grad = False

def get_bert_cls_embeddings(texts, batch_size=16):
    """Extrae SOLO el embedding del token [CLS]"""
    embeddings = []
    bert_model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Extrayendo CLS embeddings"):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LEN, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = bert_model(**encoded)
            # Extraer SOLO el primer token [CLS]
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # Shape: (batch, 768)
            embeddings.append(cls_embeddings.cpu().numpy())
    return np.vstack(embeddings)

X_train_bert_cls = get_bert_cls_embeddings(X_train)
X_test_bert_cls = get_bert_cls_embeddings(X_test)

print(f"BERT CLS embedding dim: {X_train_bert_cls.shape[1]}")

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
    grid_search.fit(X_train_bert_cls, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_bert_cls)
    
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
