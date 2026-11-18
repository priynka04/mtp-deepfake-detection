import csv, json, random, sys
from collections import defaultdict
from itertools import product
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "/DATA/storage/ab_students/align_gemini_project/Align-Dont-Blend")
from detector.core import parse_vlm_verdict

P = Path("/DATA/storage/ab_students/align_gemini_project")
GENS = ["Nano","flux-dev","flux-kera","flux-kontext","flux-pro","gpt4o","jimeng","keling","minimax"]
THRS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
GATE_THRS = [round(x, 2) for x in np.arange(0.30, 0.91, 0.05)]
SEED, TUNE_FRAC = 42, 0.30

# ---------- 1. DATA LOAD ----------
def info(path):
    p = Path(path).parts
    i = p.index("AIGI-Now")
    return p[i+1], p[i+2], int(p[i+3] == "1_fake")

def load_probs(name):
    d = {}
    for l in open(P / f"eval/aigi_all_{name}.jsonl"):
        r = json.loads(l)
        d[r["image"]] = r["fake_probability"]
    return d

def vlm_fake(resp):
    try:
        return int(bool(parse_vlm_verdict(resp)))
    except Exception:
        return 0

po, pu = load_probs("official"), load_probs("ours")
rows = []
for g in GENS:
    for l in open(P / f"eval/vlm_full/{g}.result.jsonl"):
        r = json.loads(l)
        path = r["images"][0]["path"]
        gen, sub, y = info(path)
        rows.append((gen, sub, y, vlm_fake(r["response"]), po[path], pu[path]))

GEN = np.array([r[0] for r in rows]); SUB = np.array([r[1] for r in rows])
Y = np.array([r[2] for r in rows]); V = np.array([r[3] for r in rows])
O = np.array([r[4] for r in rows]); U = np.array([r[5] for r in rows])
print("rows:", len(rows))

# ---------- 2. SAME 30/70 SPLIT AS tune_test.py ----------
random.seed(SEED)
groups = defaultdict(list)
for i, r in enumerate(rows):
    groups[(r[0], r[1], r[2])].append(i)
tune_idx = set()
for k, idxs in groups.items():
    random.shuffle(idxs)
    tune_idx.update(idxs[:int(len(idxs) * TUNE_FRAC)])
tune = np.array([i in tune_idx for i in range(len(rows))])
test = ~tune
print(f"tune={tune.sum()} test={test.sum()}")

# ---------- 3. METRICS ----------
def m(pred, y):
    tp = ((pred == 1) & (y == 1)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    fp = ((pred == 1) & (y == 0)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    return (tp + tn) / len(y), tn / max(tn + fp, 1), tp / max(tp + fn, 1)

def acc(pred, mask):
    return m(pred[mask], Y[mask])[0]

# ---------- 4. GATES ----------
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))

X = np.column_stack([V, logit(O), logit(U), V * logit(O), V * logit(U)])
FEATS = ["vlm", "off_logit", "ours_logit", "vlm*off", "vlm*ours"]

def or_pred(t):                       # OR with one threshold on official
    return ((V == 1) | (O >= t)).astype(int)

def agree_pred(a, b):                 # gate 1: both experts must agree
    return ((V == 1) | ((O >= a) & (U >= b))).astype(int)

# threshold choose karo sirf TUNE split pe
t_off = max(THRS, key=lambda t: acc(or_pred(t), tune))
a_best, b_best = max(product(THRS, THRS), key=lambda ab: acc(agree_pred(*ab), tune))

clf = LogisticRegression(max_iter=2000).fit(X[tune], Y[tune])
prob = clf.predict_proba(X)[:, 1]
g_best = max(GATE_THRS, key=lambda q: acc((prob >= q).astype(int), tune))

print(f"\nTuned on tune split -> OR official thr={t_off} | agree gate thr=(off {a_best}, ours {b_best}) | learned gate thr={g_best}")
print("Learned gate weights:", {f: round(float(c), 3) for f, c in zip(FEATS, clf.coef_[0])})

systems = {
    "vlm":           V,
    "OR@0.5":        or_pred(0.5),
    "OR@tuned":      or_pred(t_off),
    "agree_gate":    agree_pred(a_best, b_best),
    "learned_gate":  (prob >= g_best).astype(int),
}

# ---------- 5. TEST REPORT ----------
out = []
print("\n== TEST split (report these) ==")
print(f"{'system':<14}{'subset':<10}{'acc':>7}{'real':>7}{'fake':>7}")
for k, pred in systems.items():
    for sub in ("both", "pixel", "semantic"):
        mask = test if sub == "both" else test & (SUB == sub)
        a, r_, f_ = m(pred[mask], Y[mask])
        print(f"{k:<14}{sub:<10}{a:>7.3f}{r_:>7.3f}{f_:>7.3f}")
        out.append([k, sub, round(a, 4), round(r_, 4), round(f_, 4)])

print("\n== TEST per-generator acc ==")
print(f"{'gen':<13}" + "".join(f"{k:>14}" for k in systems))
for g in GENS:
    mask = test & (GEN == g)
    print(f"{g:<13}" + "".join(f"{acc(p, mask):>14.3f}" for p in systems.values()))

# ---------- 6. LEAVE-ONE-GENERATOR-OUT ----------
print("\n== Leave-one-generator-out (gate 8 generators pe seekha, 9th pe test) ==")
print(f"{'held-out':<13}{'vlm':>10}{'OR@tuned':>12}{'learned':>12}")
sums = np.zeros(3)
for g in GENS:
    te = (GEN == g); tr = ~te
    t = max(THRS, key=lambda t: acc(or_pred(t), tr))
    c = LogisticRegression(max_iter=2000).fit(X[tr], Y[tr])
    pr = c.predict_proba(X)[:, 1]
    q = max(GATE_THRS, key=lambda q: acc((pr >= q).astype(int), tr))
    a = np.array([acc(V, te), acc(or_pred(t), te), acc((pr >= q).astype(int), te)])
    sums += a
    print(f"{g:<13}{a[0]:>10.3f}{a[1]:>12.3f}{a[2]:>12.3f}")
print(f"{'MEAN':<13}" + "".join(f"{x:>{w}.3f}" for x, w in zip(sums / len(GENS), (10, 12, 12))))

# ---------- 7. CSV SAVE ----------
with open(P / "eval/gated_fusion_test.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["system", "subset", "acc", "real", "fake"]); w.writerows(out)
print("\ncsv:", P / "eval/gated_fusion_test.csv")
