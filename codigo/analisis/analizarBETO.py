import ast
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def parse_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return []
    return []

def parse_vec(x):
    if isinstance(x, str):
        try:
            return np.array(ast.literal_eval(x), dtype=float)
        except Exception:
            return np.array([])
    if isinstance(x, (list, np.ndarray)):
        return np.array(x, dtype=float)
    return np.array([])


def show_vecinos_bert_docs(bert_npz="models/bert_mean.npz", bert_index="dataset/dataset_bert.csv",
                           sample_n=10, topk=10):
    print("[1] Vecinos BERT (documentos por coseno)")
    if not Path(bert_npz).exists() or not Path(bert_index).exists():
        print(f"AVISO: Faltan {bert_npz} o {bert_index}. Omito vecinos BERT.")
        return

    idx = pd.read_csv(bert_index)
    M = np.load(bert_npz)["arr_0"]
    n = len(idx)
    if n == 0:
        print("No hay documentos en el índice.")
        return

    sample_idx = list(range(min(sample_n, n)))
    S = cosine_similarity(M[sample_idx], M)

    for a, i in enumerate(sample_idx):
        sims = S[a]
        order = np.argsort(-sims)[:topk+1]
        print(f"\n Query doc {i} | spk={idx.loc[i,'speaker']} | "
              f"id={idx.loc[i,'audio_id']} [{idx.loc[i,'start_sec']:.0f}-{idx.loc[i,'end_sec']:.0f}]")
        for r, j in enumerate(order, 1):
            print(f"   {r:2d}. doc {j:5d} | sim={sims[j]:.3f} | spk={idx.loc[j,'speaker']:<10} "
                  f"| id={idx.loc[j,'audio_id']} [{idx.loc[j,'start_sec']:.0f}-{idx.loc[j,'end_sec']:.0f}]")

# 2) PCA 2D para documentos
def show_pca_docs_2d(bert_npz="models/bert_mean.npz", bert_index="dataset/dataset_bert.csv",
                     show_n=20):
    print("[2] Proyección 2D (PCA) de documentos")
    any_shown = False

    if Path(bert_npz).exists() and Path(bert_index).exists():
        idx = pd.read_csv(bert_index)
        M = np.load(bert_npz)["arr_0"]
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(M)
        ev = pca.explained_variance_ratio_
        print(f"\n  BERT: {Z.shape[0]} documentos | Var. exp: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}")
        for i in range(min(show_n, Z.shape[0])):
            print(f"   doc {i:5d} | spk={idx.loc[i,'speaker']:<10} | pc1={Z[i,0]: .3f} | pc2={Z[i,1]: .3f}")
        any_shown = True
    else:
        print("  AVISO: No se ha encontrado bert_mean.npz/bert_index.csv")

    if not any_shown:
        print("  Faltan entradas.")


# 3) PROTOTIPOS por hablante
def show_prototipos_por_hablante(bert_npz="models/bert_mean.npz", bert_index="dataset/dataset_bert.csv",
                                 topk=5):
    print("[3] Prototipos por hablante (documentos más cercanos al centróide)")

    if Path(bert_npz).exists() and Path(bert_index).exists():
        idx = pd.read_csv(bert_index)
        M = np.load(bert_npz)["arr_0"]
        spk = idx["speaker"].values
        print("\n  Representación: BERT (mean)")
        for sp in sorted(pd.unique(spk)):
            mask = (spk == sp)
            centroid = M[mask].mean(axis=0, keepdims=True)
            sims = cosine_similarity(centroid, M[mask]).ravel()
            idx_in_class = np.where(mask)[0]
            top = idx_in_class[np.argsort(-sims)[:topk]]
            print(f"   - {sp}:")
            for j in top:
                simj = float(cosine_similarity(centroid, M[j:j+1])[0,0])
                print(f"       doc {j:5d} | sim={simj:.3f} | id={idx.loc[j,'audio_id']} "
                      f"[{idx.loc[j,'start_sec']:.0f}-{idx.loc[j,'end_sec']:.0f}]")
    else:
        print("  AVISO: No se ha encontrado bert_mean.npz/bert_index.csv.")


if __name__ == "__main__":
    semillas = ["gol", "defensa", "barcelona", "presión", "lateral"]
    show_vecinos_bert_docs(sample_n=10, topk=10)
    print()
    show_pca_docs_2d(show_n=20)
    print()
    show_prototipos_por_hablante(topk=5)
