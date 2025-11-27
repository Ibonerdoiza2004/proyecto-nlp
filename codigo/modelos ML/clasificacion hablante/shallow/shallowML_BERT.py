"""
Clasificación de Hablantes usando Shallow Machine Learning con BERT (mean pooling)
Compara 6 algoritmos: Logistic Regression, Random Forest, SVM Linear, SVM RBF, Decision Tree, Naive Bayes
Embedding: Promedio de embeddings BERT (BETO) para cada texto
Fuente: BERT como extractor de features + Shallow ML
"""

import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from time import time
from tqdm import tqdm

# Configuración
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
np.random.seed(42)
torch.manual_seed(42)

BERT_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 128
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
X_train_texts, X_test_texts, y_train, y_test = train_test_split(
    texts, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {len(X_train_texts)} muestras")
print(f"Test: {len(X_test_texts)} muestras")

# Cargar BERT
print("\n" + "="*60)
print("CARGANDO BETO PARA EXTRACCIÓN DE EMBEDDINGS")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModel.from_pretrained(BERT_MODEL)
bert_model = bert_model.to(device)
bert_model.eval()

print(f"BERT model: {BERT_MODEL}")
print(f"Embedding dimension: {bert_model.config.hidden_size}")

# Función para extraer embeddings BERT (mean pooling)
def get_bert_embeddings(texts, tokenizer, model, max_length, batch_size, device):
    """
    Extrae embeddings BERT usando mean pooling
    """
    embeddings = []
    
    # Procesar en batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Extrayendo embeddings BERT"):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenizar batch
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Obtener embeddings
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
            
            # Mean pooling (promedio de todos los tokens, ignorando padding)
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            
            embeddings.extend(mean_embeddings.cpu().numpy())
    
    return np.array(embeddings)

# Extraer embeddings para train y test
print("\n" + "="*60)
print("EXTRAYENDO EMBEDDINGS BERT (MEAN POOLING)")
print("="*60)

print("\nExtrayendo embeddings de train...")
X_train = get_bert_embeddings(X_train_texts, tokenizer, bert_model, MAX_LENGTH, BATCH_SIZE, device)

print("Extrayendo embeddings de test...")
X_test = get_bert_embeddings(X_test_texts, tokenizer, bert_model, MAX_LENGTH, BATCH_SIZE, device)

print(f"\nShape train: {X_train.shape}")
print(f"Shape test: {X_test.shape}")
print(f"Embedding dimension: {X_train.shape[1]}")

# Verificar que no haya NaN
print(f"NaN en train: {np.isnan(X_train).sum()}")
print(f"NaN en test: {np.isnan(X_test).sum()}")

# Definir clasificadores (GaussianNB para datos continuos)
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
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
    
    # Cross-validation en train
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'train_time': train_time,
        'test_time': test_time,
        'predictions': y_pred
    }
    
    trained_models[name] = clf
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Tiempo entrenamiento: {train_time:.2f}s")
    print(f"Tiempo predicción: {test_time:.4f}s")

# Comparación de resultados
print("\n" + "="*60)
print("COMPARACIÓN DE MODELOS")
print("="*60)

results_df = pd.DataFrame({
    'Modelo': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'CV Mean': [r['cv_mean'] for r in results.values()],
    'CV Std': [r['cv_std'] for r in results.values()],
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

joblib.dump(label_encoder, 'models/label_encoder_shallow.joblib')
print("Label encoder guardado en: models/label_encoder_shallow.joblib")

# Guardar todos los resultados
joblib.dump(results, 'models/results_shallow_bert.joblib')
print("Resultados guardados en: models/results_shallow_bert.joblib")

# Función de predicción
def predecir_hablante_bert(texto, tokenizer, bert_model, classifier, label_encoder, max_length, device):
    """
    Predice el hablante usando BERT + Shallow ML
    """
    # Tokenizar
    encoding = tokenizer(
        texto,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Obtener embedding BERT (mean pooling)
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        mean_embedding = (sum_embeddings / sum_mask).cpu().numpy()
    
    # Predecir
    prediccion = classifier.predict(mean_embedding)[0]
    proba = classifier.predict_proba(mean_embedding)[0] if hasattr(classifier, 'predict_proba') else None
    
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
    hablante, confianza = predecir_hablante_bert(
        ejemplo, tokenizer, bert_model, best_model, label_encoder, MAX_LENGTH, device
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
print(f"- Embedding: BERT mean pooling ({bert_model.config.hidden_size}D)")
print(f"- BERT model: {BERT_MODEL}")
print(f"- Modelos entrenados: {len(classifiers)}")
print(f"- Mejor modelo: {best_model_name}")
print(f"- Mejor accuracy: {results_df.iloc[0]['Accuracy']:.4f}")
print(f"- Tiempo total: {sum(r['train_time'] for r in results.values()):.2f}s")
print("\nVENTAJA DE BERT:")
print("- Embeddings contextuales (considera el contexto completo)")
print("- Pre-entrenado en gran corpus de español")
print("- Captura relaciones semánticas profundas")
print("- Estado del arte en representación de texto")
