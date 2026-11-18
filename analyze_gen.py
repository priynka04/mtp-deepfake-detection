import json, re, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, '/DATA/storage/ab_students/align_gemini_project/Align-Dont-Blend')
from detector.core import parse_vlm_verdict

P = Path("/DATA/storage/ab_students/align_gemini_project")
GENS = ["Nano","flux-dev","flux-kera","flux-kontext","flux-pro","gpt4o","jimeng","keling","minimax"]
T_OFF  = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
T_OURS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95

def info(path):
    p = Path(path).parts
    i = p.index("AIGI-Now")
    return p[i+1], p[i+2], int(p[i+3] == "1_fake")

def vlm_fake(resp):
    try:
        return bool(parse_vlm_verdict(resp))
    except Exception:
        return False   # unparsed -> authentic (same as eval_two_branch.py)

def load_expert(name, thr):
    d = {}
    for l in open(P / f"eval/aigi_all_{name}.jsonl"):
        r = json.loads(l)
        d[r["image"]] = r["fake_probability"] >= thr
    return d

off, ours = load_expert("official", T_OFF), load_expert("ours", T_OURS)

rows = []  # (gen, subset, label, vlm, off, ours)
miss = 0
for g in GENS:
    for l in open(P / f"eval/vlm_full/{g}.result.jsonl"):
        r = json.loads(l)
        path = r["images"][0]["path"]
        if path not in off or path not in ours:
            miss += 1; continue
        gen, sub, y = info(path)
        rows.append((gen, sub, y, vlm_fake(r["response"]), off[path], ours[path]))
print(f"rows={len(rows)} missing_expert={miss} | T_OFF={T_OFF} T_OURS={T_OURS}")

def metrics(rs, f):
    tp=tn=fp=fn=0
    for gen,sub,y,v,o,u in rs:
        p = f(v,o,u)
        if y and p: tp+=1
        elif y: fn+=1
        elif p: fp+=1
        else: tn+=1
    n = tp+tn+fp+fn
    return (tp+tn)/n, tn/max(tn+fp,1), tp/max(tp+fn,1)

systems = {
    "vlm":     lambda v,o,u: v,
    "off":     lambda v,o,u: o,
    "ours":    lambda v,o,u: u,
    "v|off":   lambda v,o,u: v or o,
    "v|ours":  lambda v,o,u: v or u,
    "v|off|ours": lambda v,o,u: v or o or u,
}

print("\n== Overall check (VLM acc should be ~0.932) ==")
for k,f in systems.items():
    print(f"{k:<11} acc/real/fake = %.3f %.3f %.3f" % metrics(rows, f))

print("\n== Per-generator (acc) ==")
print(f"{'gen':<13}" + "".join(f"{k:>11}" for k in systems))
for g in GENS:
    rs = [r for r in rows if r[0]==g]
    print(f"{g:<13}" + "".join(f"{metrics(rs,f)[0]:>11.3f}" for f in systems.values()))

print("\n== Complementarity (fake images only) ==")
for sub in ("pixel","semantic","ALL"):
    fk = [r for r in rows if r[2]==1 and (sub=="ALL" or r[1]==sub)]
    vm = [r for r in fk if not r[3]]          # VLM missed
    n, nv = len(fk), len(vm)
    print(f"[{sub}] fakes={n} | VLM missed={nv} ({nv/n:.1%})")
    print(f"   of VLM-missed: official catches {sum(r[4] for r in vm)/nv:.1%}, ours catches {sum(r[5] for r in vm)/nv:.1%}")
    ev = [r for r in fk if r[3] and not r[4]]
    print(f"   official-only gain: +{sum(1 for r in fk if (not r[3]) and r[4])} fakes, ours-only gain: +{sum(1 for r in fk if (not r[3]) and r[5])} fakes")

print("\n== Cost on real images (new false positives from OR) ==")
for sub in ("pixel","semantic","ALL"):
    rl = [r for r in rows if r[2]==0 and (sub=="ALL" or r[1]==sub)]
    ok = [r for r in rl if not r[3]]          # VLM correct on real
    print(f"[{sub}] reals={len(rl)} | extra FP from official={sum(r[4] for r in ok)}, from ours={sum(r[5] for r in ok)}")
