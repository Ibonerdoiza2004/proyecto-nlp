import torch
import torch.nn as nn
t = nn.Transformer(batch_first=True)
src = torch.randn(8, 10, 512)
tgt = torch.randn(8, 10, 512)
mask = (torch.randn(8, 10) > 0)
print("--- With nested tensor enable (default) ---")
try:
    _ = t(src, tgt, src_key_padding_mask=mask)
except Exception as e:
    print(f"Error: {e}")
print("--- Disabling nested tensor ---")
t.encoder.enable_nested_tensor = False
_ = t(src, tgt, src_key_padding_mask=mask)
