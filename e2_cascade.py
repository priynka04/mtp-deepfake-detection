import csv
import numpy as np
from common import *

recs = load_all()
tune, test = split_masks(recs)
Y = np.array([r["y"] for r in recs]); V = np.array([r["v"] for r in recs])
SC = {"official": np.array([r["o"] for r in recs]), "ours": np.array([r["u"] for r in recs])}

LOS = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
HIS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.999]
TOLS = [0.0, 0.005, 0.01, 0.02]   # tune split pe VLM se kitni accuracy girne dein

vt = acc(V, Y, tune)
print(f"VLM always: tune acc={vt:.3f} | test acc={acc(V,Y,test):.3f} | VLM calls=100%")

rows_out = []
for name, S in SC.items():
    cfgs = []
    for lo in LOS:
        for hi in HIS:
            if lo >= hi:
                continue
            sure_fake, sure_real = S >= hi, S <= lo
            call = ~(sure_fake | sure_real)                 # sirf inpe VLM chalega
            pred = np.where(sure_fake, 1, np.where(sure_real, 0, V))
            c = dict(expert=name, lo=lo, hi=hi,
                     tune_acc=acc(pred, Y, tune), tune_call=float(call[tune].mean()),
                     test_acc=acc(pred, Y, test), test_call=float(call[test].mean()))
            cfgs.append(c); rows_out.append(c)
    print(f"\n== Cascade with {name} expert (thresholds tune split pe chune, test pe report) ==")
    print(f"{'allowed drop':<13}{'lo':>7}{'hi':>7}{'test acc':>10}{'VLM calls':>11}{'saved':>8}")
    for tol in TOLS:
        ok = [c for c in cfgs if c["tune_acc"] >= vt - tol]
        if not ok:
            print(f"{tol:<13.3f}  no config"); continue
        b = min(ok, key=lambda c: c["tune_call"])           # sabse kam VLM calls
        print(f"{tol:<13.3f}{b['lo']:>7}{b['hi']:>7}{b['test_acc']:>10.3f}{b['test_call']:>11.1%}{1-b['test_call']:>8.1%}")

with open(P / "eval/e2_cascade_all.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)
print("\ncsv:", P / "eval/e2_cascade_all.csv")
