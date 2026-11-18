import csv, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, "/DATA/storage/ab_students/align_gemini_project/Align-Dont-Blend")
from detector.core import parse_vlm_verdict

P = Path("/DATA/storage/ab_students/align_gemini_project")
GENS = ["Nano", "flux-dev", "flux-kera", "flux-kontext", "flux-pro", "gpt4o", "jimeng", "keling", "minimax"]

def info(path):
    p = Path(path).parts
    i = p.index("AIGI-Now")
    return p[i+1], p[i+2], int(p[i+3] == "1_fake")

def load_expert(name, thr=0.5):
    d = {}
    for l in open(P / f"eval/aigi_all_{name}.jsonl"):
        r = json.loads(l)
        d[r["image"]] = r["fake_probability"] >= thr
    return d

T_OFF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
T_OURS = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
ex = {"official": load_expert("official", T_OFF), "ours": load_expert("ours", T_OURS)}
vlm, bad = {}, 0
for g in GENS:
    f = P / f"eval/vlm_full/{g}.result.jsonl"
    if not f.is_file():
        continue
    for l in open(f):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        v = r["images"][0]
        img = v["path"] if isinstance(v, dict) else v
        try:
            vlm[img] = parse_vlm_verdict(r["response"])
        except ValueError:
            vlm[img] = False
            bad += 1
print("VLM images:", len(vlm), "| unparsed (counted as authentic):", bad)

systems = {
    "vlm": lambda i: vlm[i],
    "expert_official": lambda i: ex["official"][i],
    "expert_ours": lambda i: ex["ours"][i],
    "or_official": lambda i: vlm[i] or ex["official"][i],
    "or_ours": lambda i: vlm[i] or ex["ours"][i],
}
cnt = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))
missing = 0
for img in vlm:
    if img not in ex["official"] or img not in ex["ours"]:
        missing += 1
        continue
    gen, sub, lab = info(img)
    for key in [(gen, sub), ("ALL", sub), ("ALL", "both")]:
        for name, fun in systems.items():
            pred = fun(img)
            c = cnt[key][name]
            if lab == 0:
                c[1] += 1
                c[0] += int(not pred)
            else:
                c[3] += 1
                c[2] += int(pred)

rows = []
for key in sorted(cnt):
    row = {"generator": key[0], "subset": key[1], "n": 0}
    for name in systems:
        ro, rn, fo, fnn = cnt[key][name]
        row["n"] = rn + fnn
        row[name + "_acc"] = round((ro + fo) / max(rn + fnn, 1), 3)
        row[name + "_real"] = round(ro / max(rn, 1), 3)
        row[name + "_fake"] = round(fo / max(fnn, 1), 3)
    rows.append(row)

out = P / "eval/two_branch_summary.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("missing expert rows:", missing)
print("system            subset    acc   real  fake")
for r in rows:
    if r["generator"] != "ALL":
        continue
    for name in systems:
        print(f"{name:<17} {r['subset']:<9} {r[name+'_acc']:.3f} {r[name+'_real']:.3f} {r[name+'_fake']:.3f}")
print("csv:", out)
