import joblib
import sys

try:
    path = 'models/best_shallow_bow.joblib'
    model = joblib.load(path)
    print(f"Type: {type(model)}")
    print(f"Model: {model}")
except Exception as e:
    print(e)
