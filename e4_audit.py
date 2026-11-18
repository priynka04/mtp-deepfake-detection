import csv, random, re, shutil
from collections import defaultdict
from common import *

CUES = {
 "anatomy":  ["hand","finger","eye","face","teeth","limb","anatomy","body","ear","arm","leg"],
 "text":     ["text","letter","writing","sign","logo","label"],
 "lighting": ["lighting","shadow","reflection","light "],
 "texture":  ["texture","smooth","plastic","skin","blur","grain","noise","sharp","detail"],
 "geometry": ["perspective","geometry","symmetr","proportion","physics","gravity","distort"],
 "context":  ["background","context","scene","object","composition"],
}
NOFLAW = re.compile(r"\b(no|not|without|lack|lacks|absence|absent|free of|consistent|natural|realistic)\b")

def rationale(resp):
    m = re.search(r"<rationale>(.*?)</rationale>", resp or "", re.S | re.I)
    return " ".join((m.group(1) if m else (resp or "")).split())

def flaw_cues(text):
    found = set()
    for s in re.split(r"(?<=[.!?])\s+", text.lower()):
        if NOFLAW.search(s):
            continue
        for cat, kws in CUES.items():
            if any(k in s for k in kws):
                found.add(cat)
    return found

def outcome(y, v):
    return ("TP" if v else "FN") if y else ("FP" if v else "TN")

recs = load_all()
by = defaultdict(list)
for i, r in enumerate(recs):
    r["rat"] = rationale(r["resp"]); r["cues"] = flaw_cues(r["rat"])
    r["oc"] = outcome(r["y"], r["v"])
    by[r["oc"]].append(i)

print("TP=fake pakda | FN=fake miss | FP=real ko fake bola | TN=real sahi\n")
print(f"{'outcome':<8}{'n':>7}{'words':>7}{'names flaw':>12}" + "".join(f"{c:>10}" for c in CUES))
for oc in ("TP", "FN", "FP", "TN"):
    rs = [recs[i] for i in by[oc]]
    n = len(rs)
    words = sum(len(r["rat"].split()) for r in rs) / n
    nf = sum(1 for r in rs if r["cues"]) / n
    print(f"{oc:<8}{n:>7}{words:>7.0f}{nf:>12.1%}" +
          "".join(f"{sum(1 for r in rs if c in r['cues'])/n:>10.1%}" for c in CUES))

# manual audit sheet + images
rng = random.Random(0)
img_dir = P / "eval/audit_images"; img_dir.mkdir(parents=True, exist_ok=True)
rows = []
k = 0
for oc in ("TP", "FN", "FP", "TN"):
    for i in rng.sample(by[oc], min(25, len(by[oc]))):
        r = recs[i]; k += 1
        fn = f"{k:03d}_{oc}.jpg"
        shutil.copy(r["path"], img_dir / fn)
        rows.append([k, oc, r["gen"], r["sub"], fn, ",".join(sorted(r["cues"])), r["rat"], "", ""])
with open(P / "eval/audit_sheet.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","outcome","generator","subset","image_file","auto_cues","rationale",
                "claimed_flaw_really_visible(Y/N/NA)","notes"])
    w.writerows(rows)
print(f"\nsheet: eval/audit_sheet.csv | images: eval/audit_images/ ({len(rows)} samples)")
