# strip_class_heads.py
import torch, re, sys
inp, out = sys.argv[1], sys.argv[2]
ckpt = torch.load(inp, map_location="cpu")
sd = ckpt["model"] if "model" in ckpt else ckpt

drop_keys = []
patterns = [
    r"class_embed",                 # any final/aux class heads
    r"enc_out_class_embed\.\d+\.",  # per-layer encoder/decoder class heads
]
for k in list(sd.keys()):
    if any(re.search(p, k) for p in patterns):
        drop_keys.append(k)

for k in drop_keys: sd.pop(k)
print(f"Dropped {len(drop_keys)} class-head params.")
torch.save({"model": sd}, out)
