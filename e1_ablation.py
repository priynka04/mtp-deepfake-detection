import csv, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *

conf = {}
for l in open(sys.argv[1]):
    r = json.loads(l); conf[r["image"]] = r["p_fake"]
recs = [r for r in load_all() if r["path"] in conf]
tune, test = split_masks(recs)
print(f"images={len(recs)} tune={tune.sum()} test={test.sum()}")
Y = np.array([r["y"] for r in recs]); PV = np.array([conf[r["path"]] for r in recs])
O = np.array([r["o"] for r in recs]); U = np.array([r["u"] for r in recs])

def logit(p, eps):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))

variants = {
    "VLM only (clip 1e-4)":          lambda: [logit(PV, 1e-4)],
    "VLM only (clip 1e-12)":         lambda: [logit(PV, 1e-12)],
    "Official expert only":          lambda: [logit(O, 1e-4)],
    "VLM + official (clip 1e-4)":    lambda: [logit(PV, 1e-4), logit(O, 1e-4)],
    "VLM + official (clip 1e-12)":   lambda: [logit(PV, 1e-12), logit(O, 1e-4)],
    "VLM + off + ours (1e-4) = E1":  lambda: [logit(PV, 1e-4), logit(O, 1e-4), logit(U, 1e-4)],
}
print(f"\n{'variant':<32}{'AUC':>7}{'acc':>7} | {'cov .90: acc':>13}{'errs flagged':>13} | {'cov .80: acc':>13}{'errs flagged':>13}")
rows = []
for name, mk in variants.items():
    X = np.column_stack(mk())
    clf = LogisticRegression(max_iter=3000).fit(X[tune], Y[tune])
    P = clf.predict_proba(X)[:, 1]; F = (P >= 0.5).astype(int)
    auc = roc_auc_score(Y[test], P[test]); acc = float((F[test] == Y[test]).mean())
    unc = np.abs(P - 0.5); out = dict(variant=name, auc=auc, acc=acc)
    line = f"{name:<32}{auc:>7.3f}{acc:>7.3f} |"
    for cov in (0.90, 0.80):
        t = np.quantile(unc[tune], 1 - cov); ans = unc >= t
        m = test & ans; err = test & (F != Y)
        a_ = float((F[m] == Y[m]).mean()); fl = (err & ~ans).sum() / max(err.sum(), 1)
        out[f"acc_cov{cov}"] = a_; out[f"flag_cov{cov}"] = fl
        line += f"{a_:>13.3f}{fl:>12.1%} |"
    print(line); rows.append(out)
with open("eval/e1_ablation.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

V = (PV > 0.5).astype(int); wrong = V != Y
def nines(p): return -np.log10(np.clip(1 - np.maximum(p, 1 - p), 1e-12, 1))
print("\n'Nines of confidence' (2 = 99%, 4 = 99.99%): median / 10th percentile")
print(f"  VLM sahi : {np.median(nines(PV)[~wrong]):.2f} / {np.percentile(nines(PV)[~wrong], 10):.2f}")
print(f"  VLM galat: {np.median(nines(PV)[wrong]):.2f} / {np.percentile(nines(PV)[wrong], 10):.2f}")
print("csv: eval/e1_ablation.csv")
