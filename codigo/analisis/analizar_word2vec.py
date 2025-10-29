from gensim.models import Word2Vec
w2v = Word2Vec.load("models/w2v.model")

def vecinos(palabra, topn=15):
    if palabra in w2v.wv.key_to_index:
        for w, s in w2v.wv.most_similar(palabra, topn=topn):
            print(f"{w:20s}  {s:.3f}")
    else:
        print(f"'{palabra}' no está en el vocabulario.")

print("Vecinos de \"gol\":")
vecinos("gol")
print("Vecinos de \"barcelona\":")
vecinos("barcelona")
print("Vecinos de \"defensa\":")
vecinos("defensa")