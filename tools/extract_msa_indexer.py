"""Extract MiniMax-M3's MSA indexer tensors from the original HF repo via byte
ranges (they were stripped from the MLX quant), and save one small sidecar
safetensors the fork can merge at load."""
import json, struct, urllib.request, sys
import numpy as np
import mlx.core as mx

REPO = "https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/main"
OUT = "/Users/phrz/.runway/models/MiniMax-M3-MLX-mixed-3_6bit/msa_indexer.safetensors"

def fetch(url, start=None, end=None):
    req = urllib.request.Request(url)
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end-1}")
    with urllib.request.urlopen(req) as r:
        return r.read()

idx = json.loads(fetch(f"{REPO}/model.safetensors.index.json").decode())
wm = idx["weight_map"]
wanted = {k: v for k, v in wm.items() if "index_" in k and "self_attn" in k}
by_shard = {}
for k, s in wanted.items():
    by_shard.setdefault(s, []).append(k)
print(f"{len(wanted)} tensors across {len(by_shard)} shards", flush=True)

DT = {"BF16": (mx.bfloat16, 2), "F16": (mx.float16, 2), "F32": (mx.float32, 4)}
out = {}
for n, (shard, keys) in enumerate(sorted(by_shard.items()), 1):
    url = f"{REPO}/{shard}"
    hlen = struct.unpack("<Q", fetch(url, 0, 8))[0]
    header = json.loads(fetch(url, 8, 8 + hlen).decode())
    base = 8 + hlen
    for k in keys:
        meta = header[k]
        dtype, ebytes = DT[meta["dtype"]]
        s, e = meta["data_offsets"]
        raw = fetch(url, base + s, base + e)
        u8 = np.frombuffer(raw, dtype=np.uint8)
        arr = mx.array(u8).view(dtype).reshape(meta["shape"])
        # language_model.model.layers.N.* -> model.layers.N.* (the fork's naming)
        out[k.replace("language_model.", "", 1)] = arr
    print(f"  shard {n}/{len(by_shard)}: +{len(keys)} tensors", flush=True)

mx.save_safetensors(OUT, out)
total = sum(v.nbytes for v in out.values())
print(f"saved {len(out)} tensors ({total/1e6:.0f} MB) -> {OUT}", flush=True)
