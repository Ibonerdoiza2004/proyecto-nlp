import numpy as np, pandas as pd, ast, joblib

df = pd.read_csv("dataset/dataset_preprocesado.csv")
df["lemmas_no_stop"] = df["lemmas_no_stop"].apply(lambda x: ast.literal_eval(x) if isinstance(x,str) else x)
texts = df["lemmas_no_stop"].apply(lambda xs: " ".join(xs)).tolist()
speakers = df["speaker"].values

tfidf_word = joblib.load("models/vec_tfidf_word.joblib")
X = tfidf_word.transform(texts)
feat = tfidf_word.get_feature_names_out()

for spk in sorted(pd.unique(speakers)):
    idx = np.where(speakers == spk)[0]
    X_mean = X[idx].mean(axis=0)
    X_mean = np.asarray(X_mean).ravel()
    top = X_mean.argsort()[-10:][::-1]
    print(f"\nTop términos para {spk}:")
    for k in top:
        print(f"  {feat[k]}: {X_mean[k]:.4f}")