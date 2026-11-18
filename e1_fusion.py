import csv, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from common import *

conf = {}
for l in open(sys.argv[1]):
    r = json.loads(l); conf[r["image"]] = r["p_fake"]
recs = [r for r in load_all() if r["path"] in conf]
print("images with VLM probability:", len(recs))
tune, test = split_masks(recs)
print(f"tune={tune.sum()} test={test.sum()}")

Y  = np.array([r["y"] for r in recs])
V  = np.array([r["v"] for r in recs])
PV = np.array([conf[r["path"]] for r in recs])
O  = np.array([r["o"] for r in recs]); U = np.array([r["u"] for r in recs])

def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
def a(pred, mask): return float((pred[mask] == Y[mask]).mean())

print("\n== 1. Signal check ==")
print(f"AUC  VLM prob: {roc_auc_score(Y[test], PV[test]):.3f} | official expert: {roc_auc_score(Y[test], O[test]):.3f}"
      f" | ours: {roc_auc_score(Y[test], U[test]):.3f}   (test split)")
sat = ((PV > 0.99) | (PV < 0.01)).mean()
wrong = (V != Y)
print(f"VLM prob 'saturated' (>0.99 ya <0.01): {sat:.1%} of images")
print(f"Jab VLM sahi: median confidence={np.median(np.maximum(PV, 1-PV)[~wrong]):.3f} | "
      f"jab VLM galat: median confidence={np.median(np.maximum(PV, 1-PV)[wrong]):.3f}")

THRS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
t_off = max(THRS, key=lambda t: a(((V == 1) | (O >= t)).astype(int), tune))
OR = ((V == 1) | (O >= t_off)).astype(int)
print(f"\n== 2. Baselines (test) ==\nVLM alone: {a(V, test):.3f} | OR@{t_off} (tuned): {a(OR, test):.3f}")

X = np.column_stack([logit(PV, 1e-12), logit(O)])   # VLM + official expert
clf = LogisticRegression(max_iter=2000).fit(X[tune], Y[tune])
P = clf.predict_proba(X)[:, 1]
print("weights [VLM-logit, official-logit]:", np.round(clf.coef_[0], 3), "| bias", np.round(clf.intercept_[0], 3))
F = (P >= 0.5).astype(int)
print(f"== 3. Learned fusion, no abstain (test) == acc={a(F, test):.3f}")

UNC = np.abs(P - 0.5)
UNC_V = np.abs(PV - 0.5)
print("\n== 4. 'Not sure' (thresholds tune split pe, report test split pe) ==")
print(f"{'target cov':>10}{'test cov':>10}{'acc answered':>14}{'OR same set':>13}{'OR on unsure':>14}{'OR errors flagged':>19}{'VLM-conf only':>15}")
rows = []
for cov in (1.0, 0.95, 0.90, 0.80, 0.70):
    t = np.quantile(UNC[tune], 1 - cov) if cov < 1 else -1
    ans = UNC >= t
    tv = np.quantile(UNC_V[tune], 1 - cov) if cov < 1 else -1
    ans_v = UNC_V >= tv
    m_ans, m_uns = test & ans, test & ~ans
    err_or = test & (OR != Y)
    flagged = (err_or & ~ans).sum() / max(err_or.sum(), 1)
    row = dict(target=cov, test_cov=m_ans.sum() / test.sum(), acc_fused=a(F, m_ans), acc_or_same=a(OR, m_ans),
               acc_or_unsure=a(OR, m_uns) if m_uns.any() else float("nan"),
               errs_flagged=flagged, acc_vlmconf=a(V, test & ans_v))
    rows.append(row)
    print(f"{cov:>10.2f}{row['test_cov']:>10.3f}{row['acc_fused']:>14.3f}{row['acc_or_same']:>13.3f}"
          f"{row['acc_or_unsure']:>14.3f}{flagged:>19.1%}{row['acc_vlmconf']:>15.3f}")
with open("eval/e1_results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("csv: eval/e1_results.csv")
