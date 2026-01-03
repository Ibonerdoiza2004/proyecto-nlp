import torch
import sys

path = 'models/clasificacion_hablantes/best_cnn_bert_finetuned.pth'
try:
    checkpoint = torch.load(path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print(f"Keys in {path}:")
    for k in list(state_dict.keys())[:10]:
        print(k)
except Exception as e:
    print(f"Error loading {path}: {e}")
