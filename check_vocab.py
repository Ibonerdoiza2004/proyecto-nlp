import joblib
try:
    vec = joblib.load('models/vec_bow.joblib')
    print(f"Vocabulary size: {len(vec.vocabulary_)}")
except Exception as e:
    print(f"Error: {e}")
