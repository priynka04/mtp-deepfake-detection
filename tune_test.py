import json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "/DATA/storage/ab_students/align_gemini_project/Align-Dont-Blend")
from detector.core import parse_vlm_verdict

P = Path("/DATA/storage/ab_students/align_gemini_project")
GENS = ["Nano","flux-dev","flux-kera","flux-kontext","flux-pro","gpt4o","jimeng","keling","minimax"]
THRS = [0.5,0.6,0.7,0.8,0.85,0.9,0.95,0.97,0.98,0.99]
SEED, TUNE_FRAC = 42, 0.30

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
        return bool(parse_vlm_verdict(resp))
    except Exception:
        return False

po, pu = load_probs("official"), load_probs("ours")
rows = []  # dict per image
for g in GENS:
    for l in open(P / f"eval/vlm_full/{g}.result.jsonl"):
        r = json.loads(l)
        path = r["images"][0]["path"]
        gen, sub, y = info(path)
        rows.append(dict(gen=gen, sub=sub, y=y, v=vlm_fake(r["response"]),
                         o=po[path], u=pu[path]))
print("rows:", len(rows))

# stratified split by (gen, subset, label)
random.seed(SEED)
groups = defaultdict(list)
for i, r in enumerate(rows):
    groups[(r["gen"], r["sub"], r["y"])].append(i)
tune_idx = set()
for k, idxs in groups.items():
    random.shuffle(idxs)
    tune_idx.update(idxs[:int(len(idxs) * TUNE_FRAC)])
tune = [r for i, r in enumerate(rows) if i in tune_idx]
test = [r for i, r in enumerate(rows) if i not in tune_idx]
print(f"tune={len(tune)} test={len(test)}")

def metrics(rs, f):
    tp=tn=fp=fn=0
    for r in rs:
        p = f(r)
        if r["y"] and p: tp+=1
        elif r["y"]: fn+=1
        elif p: fp+=1
        else: tn+=1
    return (tp+tn)/len(rs), tn/max(tn+fp,1), tp/max(tp+fn,1)

# sweep on tune split
print("\n== Tune split sweep (acc/real/fake) ==")
print(f"{'thr':<6}{'v|off':>26}{'v|ours':>26}")
best = {}
for name, key in (("off","o"), ("ours","u")):
    best[name] = max(THRS, key=lambda t: metrics(tune, lambda r: r["v"] or r[key] >= t)[0])
for t in THRS:
    a = metrics(tune, lambda r: r["v"] or r["o"] >= t)
    b = metrics(tune, lambda r: r["v"] or r["u"] >= t)
    print(f"{t:<6}{'%.3f %.3f %.3f' % a:>26}{'%.3f %.3f %.3f' % b:>26}")
T_OFF, T_OURS = best["off"], best["ours"]
print(f"\nBest on tune -> official thr={T_OFF}, ours thr={T_OURS}")

systems = {
    "vlm":            lambda r: r["v"],
    "v|off@0.5":      lambda r: r["v"] or r["o"] >= 0.5,
    "v|off@tuned":    lambda r: r["v"] or r["o"] >= T_OFF,
    "v|ours@0.5":     lambda r: r["v"] or r["u"] >= 0.5,
    "v|ours@tuned":   lambda r: r["v"] or r["u"] >= T_OURS,
}

print("\n== TEST split (report these) ==")
print(f"{'system':<14}{'subset':<10}{'acc':>7}{'real':>7}{'fake':>7}")
for k, f in systems.items():
    for sub in ("both", "pixel", "semantic"):
        rs = [r for r in test if sub == "both" or r["sub"] == sub]
        a, re_, fa = metrics(rs, f)
        print(f"{k:<14}{sub:<10}{a:>7.3f}{re_:>7.3f}{fa:>7.3f}")

print("\n== TEST per-generator acc ==")
print(f"{'gen':<13}" + "".join(f"{k:>14}" for k in systems))
for g in GENS:
    rs = [r for r in test if r["gen"] == g]
    print(f"{g:<13}" + "".join(f"{metrics(rs, f)[0]:>14.3f}" for f in systems.values()))
