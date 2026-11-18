import json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, "/DATA/storage/ab_students/align_gemini_project/Align-Dont-Blend")
from detector.core import parse_vlm_verdict

P = Path("/DATA/storage/ab_students/align_gemini_project")
GENS = ["Nano","flux-dev","flux-kera","flux-kontext","flux-pro","gpt4o","jimeng","keling","minimax"]
SEED, TUNE_FRAC = 42, 0.30

def info(path):
    p = Path(path).parts
    i = p.index("AIGI-Now")
    return p[i+1], p[i+2], int(p[i+3] == "1_fake")

def _probs(name):
    d = {}
    for l in open(P / f"eval/aigi_all_{name}.jsonl"):
        r = json.loads(l)
        d[r["image"]] = r["fake_probability"]
    return d

def _vlm(resp):
    try:
        return int(bool(parse_vlm_verdict(resp)))
    except Exception:
        return 0

def load_all():
    po, pu = _probs("official"), _probs("ours")
    recs = []
    for g in GENS:
        for l in open(P / f"eval/vlm_full/{g}.result.jsonl"):
            r = json.loads(l)
            path = r["images"][0]["path"]
            gen, sub, y = info(path)
            recs.append(dict(path=path, gen=gen, sub=sub, y=y, v=_vlm(r["response"]),
                             o=po[path], u=pu[path], resp=r["response"]))
    return recs

def split_masks(recs):
    random.seed(SEED)
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        groups[(r["gen"], r["sub"], r["y"])].append(i)
    tune_idx = set()
    for k, idxs in groups.items():
        random.shuffle(idxs)
        tune_idx.update(idxs[:int(len(idxs) * TUNE_FRAC)])
    tune = np.array([i in tune_idx for i in range(len(recs))])
    return tune, ~tune

def acc(pred, y, mask):
    return float((pred[mask] == y[mask]).mean())
