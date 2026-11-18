import os
import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss
)

from common import *

# ---------------- CONFIG ----------------

CONF_FILE = "eval/vlm_lp_s10/conf.jsonl"
OLD_BASELINE = 0.942
OUTPUT_DIR = "eval/e1_diagnostics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD DATA ----------------

conf = {}

with open(CONF_FILE) as f:
    for line in f:
        r = json.loads(line)
        conf[r["image"]] = float(r["p_fake"])

recs = [r for r in load_all() if r["path"] in conf]

print("=" * 65)
print("E1 PILOT DIAGNOSTICS")
print("=" * 65)

tune, test = split_masks(recs)

Y = np.array([r["y"] for r in recs], dtype=int)
V = np.array([r["v"] for r in recs], dtype=int)
O = np.array([r["o"] for r in recs], dtype=float)
U = np.array([r["u"] for r in recs], dtype=float)
PV = np.array([conf[r["path"]] for r in recs])

print(f"Total images : {len(recs)}")
print(f"Tune images  : {tune.sum()}")
print(f"Test images  : {test.sum()}")

# ---------------- HELPERS ----------------

def accuracy(pred, mask):
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == Y[mask]).mean())


def auc_score(prob, mask):
    if mask.sum() == 0 or len(np.unique(Y[mask])) < 2:
        return float("nan")
    return float(roc_auc_score(Y[mask], prob[mask]))


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def get_generator(rec):
    """
    Expected structure:
     .../Nano/semantic/1_fake/image.jpg

    Extracts Nano as generator.
    """
    return rec.get("gen","UNKNOWN")

def calibration_metrics(y, p, bins=10):
    """Equal-width Expected Calibration Error."""
    p = np.clip(p, 0, 1)

    brier = brier_score_loss(y, p)
    ece = 0.0
    rows = []

    edges = np.linspace(0, 1, bins + 1)

    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])

        if not mask.any():
            continue

        confidence = float(p[mask].mean())
        actual = float(y[mask].mean())
        count = int(mask.sum())

        ece += (count / len(y)) * abs(confidence - actual)

        rows.append({
            "bin": i + 1,
            "count": count,
            "mean_predicted": confidence,
            "actual_fake_rate": actual,
            "absolute_error": abs(confidence - actual)
        })

    return float(brier), float(ece), rows


# ---------------- BUILD MODELS ----------------

THRS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]

best_threshold = max(
    THRS,
    key=lambda t: accuracy(
        ((V == 1) | (O >= t)).astype(int), tune
    )
)

OR = ((V == 1) | (O >= best_threshold)).astype(int)

X = np.column_stack([
    logit(PV),
    logit(O),
    logit(U)
])

clf = LogisticRegression(max_iter=2000)
clf.fit(X[tune], Y[tune])

P = clf.predict_proba(X)[:, 1]
F = (P >= 0.5).astype(int)

print(f"\nOR threshold selected on tune: {best_threshold}")

# ==================================================
# CHECK 1: GENERATOR-WISE PERFORMANCE
# ==================================================

print("\n" + "=" * 65)
print("CHECK 1: GENERATOR-WISE PERFORMANCE")
print("=" * 65)

generators = np.array([
    get_generator(r) for r in recs
])

# Select NOT SURE threshold using tune set only.
UNC = np.abs(P - 0.5)
t80 = np.quantile(UNC[tune], 0.20)

answered = UNC >= t80

generator_rows = []

print(
    f"\n{'Generator':<18}"
    f"{'N':>7}"
    f"{'VLM Acc':>10}"
    f"{'OR Acc':>10}"
    f"{'Fusion':>10}"
    f"{'AUC':>9}"
    f"{'80% Cov':>10}"
    f"{'Sel Acc':>10}"
)

for g in sorted(set(generators[test])):
    m = test & (generators == g)
    ms = m & answered

    row = {
        "generator": g,
        "n_test": int(m.sum()),
        "vlm_accuracy": accuracy(V, m),
        "or_accuracy": accuracy(OR, m),
        "fusion_accuracy": accuracy(F, m),
        "fusion_auc": auc_score(P, m),
        "coverage": float(ms.sum() / m.sum()) if m.sum() else float("nan"),
        "selective_accuracy": accuracy(F, ms)
    }

    generator_rows.append(row)

    print(
        f"{g:<18}"
        f"{row['n_test']:>7}"
        f"{row['vlm_accuracy']:>10.3f}"
        f"{row['or_accuracy']:>10.3f}"
        f"{row['fusion_accuracy']:>10.3f}"
        f"{row['fusion_auc']:>9.3f}"
        f"{row['coverage']:>10.3f}"
        f"{row['selective_accuracy']:>10.3f}"
    )

with open(OUTPUT_DIR + "/generator_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(generator_rows[0]))
    w.writeheader()
    w.writerows(generator_rows)

# ==================================================
# CHECK 2: PROBABILITY CALIBRATION
# ==================================================

print("\n" + "=" * 65)
print("CHECK 2: PROBABILITY CALIBRATION")
print("=" * 65)

models = {
    "VLM probability": PV,
    "Learned fusion": P
}

calibration_rows = []

for name, prob in models.items():

    brier, ece, bins = calibration_metrics(
        Y[test], prob[test], bins=10
    )

    print(f"\n{name}")
    print(f"Brier Score : {brier:.5f}")
    print(f"ECE         : {ece:.5f}")

    for row in bins:
        row["model"] = name
        calibration_rows.append(row)

    print("Reliability bins:")
    print(f"{'Count':>8}{'Predicted':>14}{'Actual':>14}")

    for row in bins:
        print(
            f"{row['count']:>8}"
            f"{row['mean_predicted']:>14.4f}"
            f"{row['actual_fake_rate']:>14.4f}"
        )

with open(OUTPUT_DIR + "/calibration.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=[
        "model", "bin", "count",
        "mean_predicted", "actual_fake_rate",
        "absolute_error"
    ])
    w.writeheader()
    w.writerows(calibration_rows)

# Reliability diagram
plt.figure(figsize=(7, 7))

plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")

for name, prob in models.items():
    _, _, bins = calibration_metrics(Y[test], prob[test], bins=10)

    if bins:
        x = [r["mean_predicted"] for r in bins]
        y = [r["actual_fake_rate"] for r in bins]
        plt.plot(x, y, marker="o", label=name)

plt.xlabel("Mean predicted fake probability")
plt.ylabel("Actual fake fraction")
plt.title("E1 Reliability Diagram")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(OUTPUT_DIR + "/reliability.png", dpi=150)
plt.close()

# ==================================================
# CHECK 3: BASELINE COMPARISON
# ==================================================

print("\n" + "=" * 65)
print("CHECK 3: BASELINE COMPARISON")
print("=" * 65)

vlm_acc = accuracy(V, test)
or_acc = accuracy(OR, test)
fusion_acc = accuracy(F, test)

print(f"\nPrevious reported baseline : {OLD_BASELINE:.3%}")
print(f"Current VLM accuracy       : {vlm_acc:.3%}")
print(f"Current tuned OR accuracy  : {or_acc:.3%}")
print(f"Current learned fusion     : {fusion_acc:.3%}")

print("\nDifference from previous 94.2% reference:")

print(f"Current OR     : {(or_acc - OLD_BASELINE)*100:+.3f} percentage points")
print(f"Learned fusion : {(fusion_acc - OLD_BASELINE)*100:+.3f} percentage points")

print("\nIMPORTANT:")
print("94.2% is only a reference comparison.")
print("A fair comparison requires the old baseline predictions")
print("on exactly the same test images and labels.")

# ---------------- FINAL SUMMARY ----------------

print("\n" + "=" * 65)
print("FILES GENERATED")
print("=" * 65)

print(OUTPUT_DIR + "/generator_metrics.csv")
print(OUTPUT_DIR + "/calibration.csv")
print(OUTPUT_DIR + "/reliability.png")

print("\nDIAGNOSTICS COMPLETED.")
