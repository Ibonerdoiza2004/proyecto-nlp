class MockVocab:
    def encode(self, text):
        # Simula codificar caracteres
        return [1] * len(text)

class TextDataset:
    def __init__(self, text, vocab, seq_length):
        self.encoded = vocab.encode(text)
        print(f"Input Type: {type(text)}")
        print(f"Encoded Len: {len(self.encoded)}")
        if isinstance(text, list):
             print(f"Element 0 type: {type(text[0])}")
             # If vocab.encode iterates over the list, it effectively encodes "the list items"?
             # If MockVocab.encode does [c for c in text], iterating a list yields strings.
             
v = MockVocab()
try:
    print("--- List Input ---")
    texts = ["hola mundo", "adios mundo"]
    ds = TextDataset(texts, v, 5)
except Exception as e:
    print(e)
    
print("\n--- String Input ---")
text = "hola mundo adios mundo"
ds = TextDataset(text, v, 5)
