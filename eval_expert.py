import json, sys
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import roc_auc_score

rows = [json.loads(l) for l in open(sys.argv[1])]
g = defaultdict(list)
for r in rows:
    parts = Path(r["image"]).parts
    i = parts.index("AIGI-Now")
    gen, sub, lab = parts[i+1], parts[i+2], int(parts[i+3] == "1_fake")
    for key in [(gen, sub), ("ALL", sub), ("ALL", "both")]:
        g[key].append((lab, r["fake_probability"]))

def stats(items):
    y = [a for a, _ in items]
    p = [b for _, b in items]
    pred = [int(x >= 0.5) for x in p]
    nr = sum(1 for a in y if a == 0)
    nf = len(y) - nr
    ar = sum(1 for a, q in zip(y, pred) if a == 0 and q == 0) / max(nr, 1)
    af = sum(1 for a, q in zip(y, pred) if a == 1 and q == 1) / max(nf, 1)
    acc = sum(1 for a, q in zip(y, pred) if a == q) / len(y)
    auc = roc_auc_score(y, p) if 0 < nf < len(y) else float("nan")
    return len(y), acc, ar, af, auc

print("generator     subset    n     acc    real_acc fake_acc auc")
for k in sorted(g):
    n, acc, ar, af, auc = stats(g[k])
    print(f"{k[0]:<14}{k[1]:<10}{n:<6}{acc:.3f}  {ar:.3f}    {af:.3f}    {auc:.3f}")
