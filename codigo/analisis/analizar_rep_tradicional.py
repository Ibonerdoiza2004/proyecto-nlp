import numpy as np, pandas as pd, ast, joblib
from sklearn.metrics.pairwise import cosine_similarity

# Carga de datos y vectorizador
df = pd.read_csv("dataset/dataset_preprocesado.csv")
df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
texts = df["lemmas_no_stop"].apply(lambda xs: " ".join(xs)).tolist()
speakers = df["speaker"].values

tfidf_word = joblib.load("models/vec_tfidf_word.joblib")
X = tfidf_word.transform(texts)
feat = tfidf_word.get_feature_names_out()

# Top términos por hablante
print("\n1. Top términos por hablantes")
for spk in sorted(pd.unique(speakers)):
    idx = np.where(speakers == spk)[0]
    X_mean = X[idx].mean(axis=0)
    X_mean = np.asarray(X_mean).ravel()
    top = X_mean.argsort()[-10:][::-1]
    print(f"Top términos (TF-IDF medio) para {spk}:")
    for k in top:
        print(f"  {feat[k]}: {X_mean[k]:.4f}")
    print()

# Términos distintivos por hablante (media_hablante - media_global)
X_global_mean = np.asarray(X.mean(axis=0)).ravel()
print("\n2. Términos distintivos por hablante (media_hablante - media_global)")
for spk in sorted(pd.unique(speakers)):
    idx = np.where(speakers == spk)[0]
    X_mean_spk = np.asarray(X[idx].mean(axis=0)).ravel()
    delta = X_mean_spk - X_global_mean
    top_delta = delta.argsort()[-10:][::-1]
    print(f"Términos distintivos para {spk}:")
    for k in top_delta:
        print(f"  {feat[k]}: Δ={delta[k]:.4f} (spk={X_mean_spk[k]:.4f}, global={X_global_mean[k]:.4f})")
    print()

# Similitud entre hablantes (coseno de centroides TF-IDF)
centroids = []
names = []
for spk in sorted(pd.unique(speakers)):
    idx = np.where(speakers == spk)[0]
    c = np.asarray(X[idx].mean(axis=0)).ravel()
    centroids.append(c)
    names.append(spk)

C = np.vstack(centroids)
S = cosine_similarity(C)

print("\n3. Matriz de similitud (coseno) entre hablantes [filas/columnas en el mismo orden]:")
header = " " * 13 + " ".join(f"{n:>10s}" for n in names)
print(header)
for i, n in enumerate(names):
    row_vals = " ".join(f"{S[i,j]:10.3f}" for j in range(len(names)))
    print(f"{n:>12s} {row_vals}")

# Cargar BoW y vectorizar los mismos textos
bow = joblib.load("models/vec_bow.joblib")
X_bow = bow.transform(texts)
bow_feat = bow.get_feature_names_out()

# Top n-gramas por frecuencia
print("\n4. Top n-gramas por frecuencia")
for spk in sorted(pd.unique(speakers)):
    idx = np.where(speakers == spk)[0]
    counts = np.asarray(X_bow[idx].sum(axis=0)).ravel()
    top = counts.argsort()[-10:][::-1]
    print(f"Top n-gramas BoW para {spk} (conteos):")
    for k in top:
        print(f"  {bow_feat[k]}: {int(counts[k])}")
    print()