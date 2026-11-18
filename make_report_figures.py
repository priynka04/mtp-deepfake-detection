#!/usr/bin/env python3
"""
make_report_figures.py
Generates every figure for the detailed report and zips them.

Usage (on the server, inside a venv that has numpy, pillow, matplotlib):
  python make_report_figures.py \
      --dda_real /DATA/.../COCO-SD-2/0_real \
      --dda_fake /DATA/.../COCO-SD-2/1_fake \
      --aigi     /DATA/.../AIGI-Now \
      --out      report_figures

Every section runs independently: if one fails (wrong path etc.) the error is
printed and the rest still runs. Output: <out>/ folder + <out>.zip

NOTES ON HONESTY (please keep these in the report):
 * Sections 1-2 use REAL images from your folders.
 * Section 3 (augmentations) is OUR ILLUSTRATION of each technique with fixed
   parameters. The exact formulas of the official code (e.g. pixel_blend_mix,
   freq_blend_mix in custom_transforms.py) may differ in detail.
 * Section 6 charts use the numbers printed in your report (tables). They are
   typed into this file; if you re-run an evaluation, update them.
"""
import os, sys, random, argparse, shutil, traceback, csv
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
RNG = random.Random(42)
NP = np.random.default_rng(42)
MANIFEST = []          # (filename, description)
plt.rcParams.update({"font.size": 10, "figure.dpi": 110})


# ----------------------------------------------------------------- helpers
def list_images(folder):
    return sorted(f for f in os.listdir(folder)
                  if f.lower().endswith(EXT) and not f.startswith("."))

def load_rgb(path):
    return Image.open(path).convert("RGB")

def center_crop(img, s=336):
    w, h = img.size
    if w < s or h < s:                      # pad if smaller
        canvas = Image.new("RGB", (max(w, s), max(h, s)))
        canvas.paste(img, (0, 0))
        img = canvas
        w, h = img.size
    l, t = (w - s) // 2, (h - s) // 2
    return img.crop((l, t, l + s, t + s))

def save(fig, out, name, desc):
    path = os.path.join(out, name)
    fig.savefig(path, bbox_inches="tight", dpi=170)
    plt.close(fig)
    MANIFEST.append((name, desc))
    print("  saved", name)

def run(title, fn, *a):
    print(f"\n== {title}")
    try:
        fn(*a)
    except Exception:
        print("  !! FAILED (rest continues):")
        traceback.print_exc()


# ------------------------------------------------ 1. DDA training pairs
def sec_dda_pairs(real_dir, fake_dir, out, n=4):
    reals = set(list_images(real_dir)); fakes = set(list_images(fake_dir))
    common = sorted(reals & fakes)
    print(f"  {len(reals)} real, {len(fakes)} fake, {len(common)} matched names")
    pick = RNG.sample(common, n)
    fig, ax = plt.subplots(n, 4, figsize=(13, 3.2 * n))
    for r, name in enumerate(pick):
        a = np.asarray(center_crop(load_rgb(os.path.join(real_dir, name))), dtype=np.float32)
        b = np.asarray(center_crop(load_rgb(os.path.join(fake_dir, name))), dtype=np.float32)
        d = np.abs(a - b).mean(axis=2)
        d_amp = np.clip(d / (np.percentile(d, 99.5) + 1e-6), 0, 1)
        ax[r, 0].imshow(a.astype(np.uint8)); ax[r, 1].imshow(b.astype(np.uint8))
        ax[r, 2].imshow(d_amp, cmap="inferno")
        ax[r, 3].hist(d.ravel(), bins=60, color="tab:purple")
        ax[r, 3].set_yscale("log"); ax[r, 3].set_xlabel("|real - fake| per pixel")
        if r == 0:
            ax[r, 0].set_title("REAL (COCO photo)")
            ax[r, 1].set_title("FAKE twin (VAE reconstruction)")
            ax[r, 2].set_title("Difference, amplified\n(what the network must find)")
            ax[r, 3].set_title("How small the difference is")
        for c in range(3): ax[r, c].axis("off")
        ax[r, 0].text(0, -8, name, fontsize=7, va="bottom")
    fig.suptitle("DDA training pairs: same content, different low-level traces", y=1.0)
    save(fig, out, "01_dda_pairs.png",
         "Real/fake training pairs + amplified difference (Section: dataset)")

    # average spectrum real vs fake
    k = min(60, len(common)); acc_r = acc_f = 0
    for name in RNG.sample(common, k):
        for src, which in ((real_dir, "r"), (fake_dir, "f")):
            g = np.asarray(center_crop(load_rgb(os.path.join(src, name)), 256)
                           .convert("L"), dtype=np.float32)
            F = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))
            if which == "r": acc_r = acc_r + F
            else: acc_f = acc_f + F
    acc_r /= k; acc_f /= k
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a_, im, t in ((ax[0], acc_r, "Average spectrum: REAL"),
                      (ax[1], acc_f, "Average spectrum: FAKE twin")):
        a_.imshow(im, cmap="magma"); a_.set_title(t); a_.axis("off")
    dd = acc_f - acc_r
    v = np.percentile(np.abs(dd), 99)
    ax[2].imshow(dd, cmap="coolwarm", vmin=-v, vmax=v)
    ax[2].set_title("FAKE minus REAL (frequency fingerprint)"); ax[2].axis("off")
    fig.suptitle(f"Frequency view over {k} pairs (log |FFT|, grayscale 256x256 crop)")
    save(fig, out, "02_dda_spectrum.png",
         "Average Fourier spectrum real vs fake: shows generator 'fingerprint'")


# ------------------------------------------------ 2. AIGI-Now structure
def sec_aigi(root, out):
    gens = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and not d.startswith("."))
    subsets = ["pixel", "semantic"]; labels = [("0_real", "real"), ("1_fake", "fake")]
    counts = {}
    for g in gens:
        for s in subsets:
            for l, _ in labels:
                p = os.path.join(root, g, s, l)
                counts[(g, s, l)] = len(list_images(p)) if os.path.isdir(p) else 0
    # CSV + bar chart
    with open(os.path.join(out, "aigi_counts.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["generator", "subset", "label", "count"])
        for (g, s, l), c in counts.items(): w.writerow([g, s, l, c])
    MANIFEST.append(("aigi_counts.csv", "Image counts per generator/subset/label"))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(gens)); wd = 0.2
    series = [("pixel", "0_real"), ("pixel", "1_fake"), ("semantic", "0_real"), ("semantic", "1_fake")]
    cols = ["#9ecae1", "#3182bd", "#fdae6b", "#e6550d"]
    for i, (s, l) in enumerate(series):
        ax.bar(x + (i - 1.5) * wd, [counts[(g, s, l)] for g in gens], wd,
               label=f"{s} / {l.split('_')[1]}", color=cols[i])
    ax.set_xticks(x); ax.set_xticklabels(gens, rotation=20)
    ax.set_ylabel("number of images"); ax.legend(ncol=4)
    ax.set_title(f"AIGI-Now structure: {sum(counts.values()):,} images, 9 generators x 2 subsets x real/fake")
    save(fig, out, "03_aigi_counts.png", "AIGI-Now image counts per generator")

    # big sample grid: one row per generator, 4 columns
    fig, ax = plt.subplots(len(gens), 4, figsize=(11, 2.6 * len(gens)))
    heads = ["pixel / REAL", "pixel / FAKE", "semantic / REAL", "semantic / FAKE"]
    for r, g in enumerate(gens):
        c = 0
        for s in subsets:
            for l, _ in labels:
                p = os.path.join(root, g, s, l)
                a = ax[r, c]; a.axis("off")
                if os.path.isdir(p) and list_images(p):
                    im = load_rgb(os.path.join(p, RNG.choice(list_images(p))))
                    im.thumbnail((320, 320)); a.imshow(im)
                if r == 0: a.set_title(heads[c], fontsize=10)
                if c == 0: a.text(-0.05, 0.5, g, transform=a.transAxes, ha="right",
                                  va="center", fontsize=10, fontweight="bold")
                c += 1
    fig.suptitle("One random sample per generator, subset and label", y=1.0)
    save(fig, out, "04_aigi_sample_grid.png", "AIGI-Now: sample from every generator")

    # per-generator gallery
    gdir = os.path.join(out, "aigi_per_generator"); os.makedirs(gdir, exist_ok=True)
    for g in gens:
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        row = 0
        for s in subsets:
            for l, ln in labels:
                p = os.path.join(root, g, s, l)
                imgs = list_images(p) if os.path.isdir(p) else []
                picks = RNG.sample(imgs, min(4, len(imgs)))
                for c in range(4):
                    a = ax[row, c]; a.axis("off")
                    if c < len(picks):
                        im = load_rgb(os.path.join(p, picks[c])); im.thumbnail((300, 300)); a.imshow(im)
                ax[row, 0].set_title(f"{s} / {ln}", loc="left", fontsize=10)
                row += 1
        fig.suptitle(f"Generator: {g}", y=0.93)
        fig.savefig(os.path.join(gdir, f"{g}.png"), bbox_inches="tight", dpi=130); plt.close(fig)
    MANIFEST.append(("aigi_per_generator/<generator>.png", "4 samples per subset/label for each generator"))
    print("  saved aigi_per_generator/*.png")


# ------------------------------------------------ 3. augmentations (illustration)
def jpeg(img, q):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")

def gauss_noise(img, sigma=15):
    a = np.asarray(img, dtype=np.float32) + NP.normal(0, sigma, np.asarray(img).shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

def pepper_noise(img, p=0.03):
    a = np.asarray(img).copy(); m = NP.random(a.shape[:2]) < p; a[m] = 0
    return Image.fromarray(a)

def motion_blur(img):
    k = ImageFilter.Kernel((5, 5), [0]*10 + [1]*5 + [0]*10, scale=5)
    return img.filter(k)

def hue_shift(img, amount=0.175):
    h, s, v = img.convert("HSV").split()
    h = h.point(lambda x: int((x + amount * 255) % 256))
    return Image.merge("HSV", (h, s, v)).convert("RGB")

def color_jitter(img):
    img = ImageEnhance.Brightness(img).enhance(1.4)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Color(img).enhance(1.4)
    return hue_shift(img, 0.175)

def pixel_mix(real, fake, ratio=0.5):
    return Image.blend(real, fake, ratio)

def freq_mix(real, fake, radius_frac=0.12):
    a = np.asarray(real, dtype=np.float32); b = np.asarray(fake, dtype=np.float32)
    h, w = a.shape[:2]; yy, xx = np.ogrid[:h, :w]
    mask = ((yy - h/2)**2 + (xx - w/2)**2) <= (radius_frac * min(h, w))**2
    out = np.zeros_like(a)
    for c in range(3):
        Fa = np.fft.fftshift(np.fft.fft2(a[..., c])); Fb = np.fft.fftshift(np.fft.fft2(b[..., c]))
        Fm = np.where(mask, Fa, Fb)           # low freq from real, high freq from fake
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(Fm)))
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def resize_to(img, f, size=336):
    w, h = img.size
    r = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BICUBIC)
    return r

def sec_aug(real_dir, fake_dir, out):
    name = RNG.choice(sorted(set(list_images(real_dir)) & set(list_images(fake_dir))))
    real = center_crop(load_rgb(os.path.join(real_dir, name)))
    fake = center_crop(load_rgb(os.path.join(fake_dir, name)))
    adir = os.path.join(out, "augmentations_individual"); os.makedirs(adir, exist_ok=True)

    items = [
        ("Original FAKE (input)", fake, "-"),
        ("JPEG matching\n(fake gets real's quality)", jpeg(fake, 70), "p=0.5 (DDA step)"),
        ("Pixel-mix\n(blend real+fake, illustration)", pixel_mix(real, fake, 0.5), "p=0.2 (DDA step)"),
        ("Frequency-mix\n(low freq real, high freq fake)", freq_mix(real, fake), "p=0.2 (DDA step)"),
        ("Random JPEG", jpeg(fake, 55), "quality 55-100, p=0.15"),
        ("Horizontal flip", ImageOps.mirror(fake), "p=0.5"),
        ("Vertical flip", ImageOps.flip(fake), "p=0.5"),
        ("Gaussian noise", gauss_noise(fake), "p=0.1"),
        ("Pepper noise", pepper_noise(fake), "p=0.1"),
        ("Gaussian blur", fake.filter(ImageFilter.GaussianBlur(2)), "blur group p=0.2"),
        ("Median blur (3)", fake.filter(ImageFilter.MedianFilter(3)), "blur group p=0.2"),
        ("Motion blur (5)", motion_blur(fake), "blur group p=0.2"),
        ("Sharpen", ImageEnhance.Sharpness(fake).enhance(3.0), "p=0.1"),
        ("Colour jitter", color_jitter(fake), "0.4/0.4/0.4/0.175, p=0.2"),
        ("Random grayscale", ImageOps.grayscale(fake).convert("RGB"), "p=0.1"),
        ("Random crop 336\n(after pad)", fake.crop((40, 40, 336, 336)).resize((336, 336)), "always"),
    ]
    cols = 4; rows = int(np.ceil(len(items) / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.6 * rows))
    for a in ax.ravel(): a.axis("off")
    for i, (t, im, setting) in enumerate(items):
        a = ax.ravel()[i]; a.imshow(im); a.set_title(f"{t}\n[{setting}]", fontsize=8.5)
        safe = "".join(ch if ch.isalnum() else "_" for ch in t.split("\n")[0])
        im.save(os.path.join(adir, f"{i:02d}_{safe}.png"))
    fig.suptitle("Augmentation techniques used in Branch 2 training (our illustration, same source image)", y=1.0)
    save(fig, out, "05_augmentations.png", "Every augmentation applied to one image")
    MANIFEST.append(("augmentations_individual/*.png", "Each augmentation as a separate image"))

    # the 6 images of one training sample
    six = [("1. REAL", real), ("2. FAKE", fake),
           ("3. REAL, resized x0.4", resize_to(real, 0.4)), ("4. REAL, resized x2.0", resize_to(real, 2.0)),
           ("5. FAKE, resized x0.4", resize_to(fake, 0.4)), ("6. FAKE, resized x2.0", resize_to(fake, 2.0))]
    fig, ax = plt.subplots(1, 6, figsize=(19, 3.6))
    for a, (t, im) in zip(ax, six):
        a.imshow(im); a.set_title(f"{t}\n{im.size[0]}x{im.size[1]} px", fontsize=9); a.axis("off")
    fig.suptitle("One training SAMPLE = 6 images (real, fake, two resized reals, two resized fakes)", y=1.04)
    save(fig, out, "06_six_images_per_sample.png", "The 6 images made from one training pair")

    # ViT patch grid
    img = real
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.3), gridspec_kw={"width_ratios": [3, 1.2]})
    ax[0].imshow(img)
    for k in range(0, 337, 14):
        ax[0].axhline(k - 0.5, color="white", lw=0.35, alpha=0.8)
        ax[0].axvline(k - 0.5, color="white", lw=0.35, alpha=0.8)
    ax[0].add_patch(Rectangle((14*10 - .5, 14*8 - .5), 14, 14, fill=False, ec="red", lw=2))
    ax[0].set_title("336x336 crop cut into 24 x 24 = 576 patches of 14x14 pixels"); ax[0].axis("off")
    ax[1].imshow(img.crop((140, 112, 154, 126)).resize((280, 280), Image.NEAREST))
    ax[1].set_title("one patch (14x14 px)\nzoomed 20x = one 'word' for the transformer"); ax[1].axis("off")
    save(fig, out, "07_vit_patches.png", "How the ViT sees the image: 576 patches + CLS")


# ------------------------------------------------ 4. diagrams
def _box(ax, x, y, w, h, text, fc="#dbe9f6", ec="#2b5c8a", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)

def _arrow(ax, x1, y1, x2, y2, text=None, color="#333", fs=8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=1.4))
    if text:
        ax.text((x1 + x2)/2, (y1 + y2)/2 + 0.12, text, ha="center", fontsize=fs, color=color)

def _canvas(w, h, title=None, size=(13, 6)):
    fig, ax = plt.subplots(figsize=size); ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    if title: ax.set_title(title, fontsize=12, fontweight="bold")
    return fig, ax

GREY, GREEN, ORANGE, BLUE, RED = "#e6e6e6", "#d5efd5", "#fde2c3", "#dbe9f6", "#f6d6d6"

def sec_diagrams(out):
    # 4.1 overall project pipeline
    fig, ax = _canvas(14, 3.4, "Project roadmap: what we did, step by step", (14, 4))
    steps = ["1. Setup\nGPU, 3 venvs", "2. Data\nDDA (train)\nAIGI-Now (test)", "3. Train\nBranch 2\n(17.4 h)",
             "4. Run\nBranch 1 VLM\n(3 h 46 m)", "5. Evaluate\nboth branches", "6. Fuse\nOR rule +\nthreshold",
             "7. Extensions\ngate, cascade,\naudit", "8. Dashboard\n+ live tests"]
    for i, t in enumerate(steps):
        x = 0.1 + i * 1.74
        _box(ax, x, 1.0, 1.5, 1.6, t, fc=[BLUE, GREY, GREEN, ORANGE, BLUE, ORANGE, GREEN, GREY][i], fs=8.5)
        if i < len(steps) - 1: _arrow(ax, x + 1.5, 1.8, x + 1.74, 1.8)
    save(fig, out, "10_roadmap.png", "Project roadmap")

    # 4.2 two-branch system
    fig, ax = _canvas(14, 6, "Two-branch detector (AlignGemini)")
    _box(ax, 0.2, 2.5, 1.8, 1.0, "Input\nimage", GREY, bold=True)
    _box(ax, 3.2, 4.0, 3.6, 1.3, "BRANCH 1: semantic\nQwen2.5-VL-7B + LoRA\n'Does the scene make sense?'", ORANGE, bold=True)
    _box(ax, 3.2, 0.8, 3.6, 1.3, "BRANCH 2: pixel expert\nDINOv2 ViT-L/14 + LoRA\n'Are the pixels natural?'", GREEN, bold=True)
    _box(ax, 8.0, 4.0, 2.4, 1.3, "Verdict (real/fake)\n+ written reason", "#fff3df")
    _box(ax, 8.0, 0.8, 2.4, 1.3, "Fake probability\np in [0, 1]", "#e6f5e6")
    _box(ax, 11.3, 2.4, 1.6, 1.2, "FUSION\nOR rule", BLUE, bold=True)
    _arrow(ax, 2.0, 3.0, 3.2, 4.6); _arrow(ax, 2.0, 3.0, 3.2, 1.5)
    _arrow(ax, 6.8, 4.65, 8.0, 4.65); _arrow(ax, 6.8, 1.45, 8.0, 1.45)
    _arrow(ax, 10.4, 4.65, 11.5, 3.6); _arrow(ax, 10.4, 1.45, 11.5, 2.4)
    ax.text(12.1, 1.7, "fake if EITHER\nbranch says fake", ha="center", fontsize=9)
    save(fig, out, "11_two_branch_system.png", "Two-branch system diagram")

    # 4.3 why two branches (data-driven chart)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    labs = ["pixel subset\n(artifacts visible)", "semantic subset\n(artifacts in meaning)"]
    x = np.arange(2); w = 0.26
    ax[0].bar(x - w, [.880, .958], w, label="VLM (semantic branch)", color="#e6550d")
    ax[0].bar(x, [.677, .147], w, label="Official pixel expert", color="#31a354")
    ax[0].bar(x + w, [.778, .473], w, label="Our pixel expert", color="#9ecae1")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labs); ax[0].set_ylim(0, 1.05)
    ax[0].set_ylabel("share of FAKE images caught (fake accuracy)")
    ax[0].set_title("Each branch is strong on a different kind of fake"); ax[0].legend(fontsize=8)
    for i, vals in enumerate([[.880, .677, .778], [.958, .147, .473]]):
        for j, v in enumerate(vals):
            ax[0].text(i + (j - 1) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax[1].bar(["VLM only", "Pixel expert only\n(official)", "Both (OR)"], [.932, .695, .939],
              color=["#e6550d", "#31a354", "#3182bd"])
    ax[1].set_ylim(0.6, 1.0); ax[1].set_ylabel("accuracy, all 36,000 images")
    ax[1].set_title("What happens if one branch is missing?")
    for i, v in enumerate([.932, .695, .939]): ax[1].text(i, v + .005, f"{v:.3f}", ha="center")
    save(fig, out, "12_why_two_branches.png", "Why two branches / what if one fails (from results)")

    # 4.4 ViT block with LoRA positions
    fig, ax = _canvas(13, 6.2, "Inside ONE of the 24 transformer blocks: where LoRA is attached", (13, 6.4))
    _box(ax, 0.3, 2.6, 1.5, 1.0, "tokens in\n(577 x 1024)", GREY)
    _box(ax, 2.6, 2.6, 1.2, 1.0, "LayerNorm", "#eeeeee")
    _box(ax, 4.5, 3.9, 2.0, 0.9, "attn.qkv  + LoRA", GREEN, bold=True)
    _box(ax, 4.5, 2.6, 2.0, 0.9, "self-attention", BLUE)
    _box(ax, 4.5, 1.3, 2.0, 0.9, "attn.proj + LoRA", GREEN, bold=True)
    _box(ax, 7.2, 2.6, 1.2, 1.0, "+ residual", "#eeeeee")
    _box(ax, 9.0, 3.9, 2.0, 0.9, "mlp.fc1 + LoRA", GREEN, bold=True)
    _box(ax, 9.0, 2.6, 2.0, 0.9, "GELU", BLUE)
    _box(ax, 9.0, 1.3, 2.0, 0.9, "mlp.fc2 + LoRA", GREEN, bold=True)
    _box(ax, 11.6, 2.6, 1.2, 1.0, "+ residual\n(out)", "#eeeeee")
    for (a, b) in [(1.8, 2.6), (3.8, 4.5), (6.5, 7.2), (8.4, 9.0), (11.0, 11.6)]:
        _arrow(ax, a, 3.1, b, 3.1)
    ax.text(6.5, 0.4, "Green = the 4 linear layers that get a LoRA add-on (all 24 blocks).  Grey/blue = frozen DINOv2.\n"
                      "Only the LoRA matrices (and possibly the 1025-parameter head) are trained.", ha="center", fontsize=9.5)
    save(fig, out, "13_vit_block_lora.png", "Transformer block with LoRA attachment points")

    # 4.5 LoRA math
    fig, ax = _canvas(12, 5, "LoRA: a small correction added to a frozen layer", (11, 5))
    _box(ax, 0.3, 2.0, 1.4, 1.0, "input x", GREY)
    _box(ax, 3.2, 3.2, 3.0, 1.0, "W  (frozen, big)\n1024 x 1024 ...", "#d9d9d9", bold=True)
    _box(ax, 3.2, 0.9, 1.3, 1.0, "A\n(r x in)", GREEN, bold=True)
    _box(ax, 4.9, 0.9, 1.3, 1.0, "B\n(out x r)", GREEN, bold=True)
    _box(ax, 7.2, 2.0, 1.2, 1.0, "+", BLUE, bold=True)
    _box(ax, 9.4, 2.0, 1.6, 1.0, "output y", GREY)
    _arrow(ax, 1.7, 2.5, 3.2, 3.7); _arrow(ax, 1.7, 2.5, 3.2, 1.4)
    _arrow(ax, 4.5, 1.4, 4.9, 1.4); _arrow(ax, 6.2, 3.7, 7.2, 2.7); _arrow(ax, 6.2, 1.4, 7.2, 2.3)
    _arrow(ax, 8.4, 2.5, 9.4, 2.5)
    ax.text(5.5, 0.2, r"y = W x + (alpha / r) * B A x ,   r = 8,  alpha = 1   (B starts at 0, so at step 0 y = W x)",
            ha="center", fontsize=10)
    ax.text(8.0, 4.4, "Only A and B learn.\nW never changes.", ha="center", fontsize=10, color="#2b7a2b")
    save(fig, out, "14_lora.png", "LoRA diagram")

    # 4.6 training diagram
    fig, ax = _canvas(16, 9, "Branch 2 training: what happens in ONE training step", (16, 9.4))
    _box(ax, 0.2, 7.2, 2.2, 1.2, "DDA pair from disk\nREAL.png + FAKE.png", GREY, bold=True)
    _box(ax, 3.0, 7.0, 4.6, 1.6, "Loader (datasets.py)\n(1) JPEG-match fake  p=0.5\n(2) pixel-mix p=0.2, freq-mix p=0.2\n(3) make 2 resized copies each", ORANGE, fs=8.5)
    _box(ax, 8.2, 7.2, 3.0, 1.2, "6 images per sample\n+ standard augmentation\n+ crop 336x336", ORANGE, fs=8.5)
    _box(ax, 11.8, 7.2, 3.9, 1.2, "2 samples per step = 12 images\n(label: real=0, fake=1)", GREY, fs=9)
    _arrow(ax, 2.4, 7.8, 3.0, 7.8); _arrow(ax, 7.6, 7.8, 8.2, 7.8); _arrow(ax, 11.2, 7.8, 11.8, 7.8)
    _box(ax, 0.6, 4.6, 3.4, 1.4, "DINOv2 ViT-L/14\nFROZEN (24 blocks)", "#d9d9d9", bold=True)
    _box(ax, 0.6, 3.0, 3.4, 1.1, "LoRA add-ons (r=8)\nTRAINED  (~3.1 M numbers)", GREEN, bold=True)
    _box(ax, 4.8, 4.2, 2.4, 1.2, "CLS vector\n(1024 numbers)", BLUE)
    _box(ax, 7.8, 4.2, 2.4, 1.2, "Linear head\n1024 -> 1  (logit)", BLUE)
    _box(ax, 10.8, 5.0, 2.4, 1.0, "BCE loss\n(real vs fake)", RED)
    _box(ax, 10.8, 3.6, 2.4, 1.0, "Contrastive loss\n(pull same class together)", RED)
    _box(ax, 13.7, 4.2, 2.0, 1.2, "loss =\n0.5 BCE + 0.5 contr.", RED, bold=True)
    _arrow(ax, 13.5, 7.2, 3.0, 6.0, "12 images in")
    _arrow(ax, 2.3, 4.1, 2.3, 4.6, "plugged into the blocks", "#2b7a2b")
    _arrow(ax, 4.0, 5.3, 4.8, 4.8); _arrow(ax, 7.2, 4.8, 7.8, 4.8)
    _arrow(ax, 10.2, 4.8, 10.8, 5.4); _arrow(ax, 10.2, 4.7, 10.8, 4.1)
    _arrow(ax, 13.2, 5.4, 13.7, 4.9); _arrow(ax, 13.2, 4.1, 13.7, 4.7)
    _box(ax, 3.0, 0.8, 4.2, 1.5, "Backward pass:\ngradients only for LoRA (+head)\nloss is divided by 4", GREEN, fs=9)
    _box(ax, 7.9, 0.8, 3.4, 1.5, "Accumulate 4 steps\n= 8 samples = 48 images\nthen ONE optimizer step", ORANGE, fs=9)
    _box(ax, 12.0, 0.8, 3.7, 1.5, "AdamW, lr 1e-4\ncosine schedule\nrepeat for 59,143 steps (1 epoch)", BLUE, fs=9)
    _arrow(ax, 14.7, 4.2, 5.1, 2.3); _arrow(ax, 7.2, 1.55, 7.9, 1.55); _arrow(ax, 11.3, 1.55, 12.0, 1.55)
    _arrow(ax, 13.9, 2.3, 2.3, 3.0, "updated LoRA values go back into the model", "#2b7a2b")
    save(fig, out, "15_training_diagram.png", "Architecture-level diagram of the training")

    # 4.7 Branch 1 VLM
    fig, ax = _canvas(14, 5.6, "Branch 1: how the VLM judges one image", (14, 5.6))
    _box(ax, 0.2, 3.5, 1.9, 1.0, "Image", GREY, bold=True)
    _box(ax, 0.2, 1.6, 1.9, 1.3, "Prompt:\n'rationale < 120 words,\nsemantic reasons only,\nthen answer'", GREY, fs=8)
    _box(ax, 3.0, 2.4, 3.2, 1.5, "Qwen2.5-VL-7B\n(vision encoder + language model)\nFROZEN", "#d9d9d9", bold=True)
    _box(ax, 3.0, 0.5, 3.2, 1.0, "LoRA adapter (authors')\nrank 16, language model only", GREEN, fs=9)
    _box(ax, 7.1, 3.2, 3.1, 1.0, "<rationale> written reason", "#fff3df")
    _box(ax, 7.1, 1.7, 3.1, 1.0, "<answer> authentic / AI-generated", "#fff3df")
    _box(ax, 11.1, 2.4, 2.6, 1.3, "official parser\n-> real / fake", BLUE, bold=True)
    _arrow(ax, 2.1, 4.0, 3.0, 3.4); _arrow(ax, 2.1, 2.3, 3.0, 2.8); _arrow(ax, 4.6, 1.5, 4.6, 2.4)
    _arrow(ax, 6.2, 3.4, 7.1, 3.7); _arrow(ax, 6.2, 2.9, 7.1, 2.2)
    _arrow(ax, 10.2, 2.2, 11.1, 2.8)
    save(fig, out, "16_vlm_branch.png", "Branch 1 VLM flow")

    # 4.8 fusion + cascade
    fig, ax = _canvas(14, 6.4, "Fusion rules and the cascade", (14, 6.4))
    _box(ax, 0.2, 4.6, 6.3, 1.3, "OR rule (paper):\nfake = VLM says fake  OR  expert score >= threshold", BLUE, bold=True)
    _box(ax, 0.2, 2.9, 6.3, 1.3, "Gated fusion (ours):\nlisten to the expert only if both experts agree (agreement gate)\nor if a small logistic regression says so (learned gate)", GREEN, fs=9)
    ax.text(3.3, 2.1, "Result: no gain over OR with a tuned threshold (0.941-0.942 vs 0.942)", ha="center", fontsize=9, color="#a33")
    ax.text(10.4, 5.95, "Cascade (E2): cheap expert first, VLM only if unsure", ha="center", fontweight="bold")
    ax.add_patch(Rectangle((7.2, 4.5), 1.6, 0.6, fc="#c7e9c0", ec="k")); ax.text(8.0, 4.8, "S <= lo", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((8.8, 4.5), 2.6, 0.6, fc="#fdd0a2", ec="k")); ax.text(10.1, 4.8, "lo < S < hi", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((11.4, 4.5), 1.6, 0.6, fc="#fcbba1", ec="k")); ax.text(12.2, 4.8, "S >= hi", ha="center", va="center", fontsize=9)
    ax.text(8.0, 4.2, "answer REAL\n(no VLM call)", ha="center", fontsize=8.5)
    ax.text(10.1, 4.2, "call the VLM", ha="center", fontsize=8.5)
    ax.text(12.2, 4.2, "answer FAKE\n(no VLM call)", ha="center", fontsize=8.5)
    ax.text(10.1, 3.1, "S = pixel-expert fake probability", ha="center", fontsize=9)
    ax.text(10.1, 2.2, "Result: about 22-24 % of VLM calls saved\nwith no accuracy loss (0.936-0.939 vs 0.933)", ha="center", fontsize=9, color="#2b7a2b")
    _box(ax, 3.0, 0.2, 8.0, 1.2, "E4 audit: read all 36,000 written reasons, count which flaw type\n(anatomy, text, lighting, texture, geometry, context) is named", ORANGE, fs=9)
    save(fig, out, "17_fusion_cascade_audit.png", "Fusion / gating / cascade / audit explained")

    # 4.9 trainable vs frozen
    r = 8; per_block = r*(1024+3072) + r*(1024+1024) + r*(1024+4096) + r*(4096+1024)
    lora = per_block * 24; head = 1024 + 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(["DINOv2 backbone\n(frozen)", "LoRA\n(trained)"], [304e6, lora], color=["#bdbdbd", "#31a354"])
    ax[0].set_yscale("log"); ax[0].set_ylabel("number of parameters (log scale)")
    ax[0].set_title("Only ~1% of the numbers are trained")
    ax[0].text(0, 304e6 * 1.2, "~304 M", ha="center"); ax[0].text(1, lora * 1.2, f"{lora:,}", ha="center")
    ax[1].axis("off")
    ax[1].text(0, 0.95, "LoRA parameters (calculated, rank 8):", fontweight="bold", va="top")
    lines = [f"attn.qkv : 8 x (1024 + 3072) = {r*(1024+3072):,}",
             f"attn.proj: 8 x (1024 + 1024) = {r*(1024+1024):,}",
             f"mlp.fc1  : 8 x (1024 + 4096) = {r*(1024+4096):,}",
             f"mlp.fc2  : 8 x (4096 + 1024) = {r*(4096+1024):,}",
             f"per block: {per_block:,}", f"x 24 blocks = {lora:,}",
             f"+ linear head (if trained): {head:,}",
             "", "Exact count: run count_params.py on the checkpoint."]
    ax[1].text(0, 0.85, "\n".join(lines), family="monospace", va="top", fontsize=9)
    save(fig, out, "18_trainable_parameters.png", "Trainable vs frozen parameters (calculated)")


# ------------------------------------------------ 5. result charts (numbers from the report)
GEN = ["Nano", "flux-dev", "flux-kera", "flux-kontext", "flux-pro", "gpt4o", "jimeng", "keling", "minimax"]

def sec_charts(out):
    # per generator VLM vs OR official
    vlm = [.961, .955, .937, .845, .904, .965, .953, .960, .905]
    orr = [.958, .954, .929, .883, .926, .961, .957, .956, .925]
    fig, ax = plt.subplots(figsize=(11, 4.4)); x = np.arange(9)
    ax.bar(x - .2, vlm, .4, label="VLM alone", color="#e6550d")
    ax.bar(x + .2, orr, .4, label="VLM OR official expert", color="#3182bd")
    ax.set_ylim(0.8, 1.0); ax.set_xticks(x); ax.set_xticklabels(GEN, rotation=20)
    ax.set_ylabel("accuracy (4,000 images per generator)"); ax.legend()
    ax.set_title("Results on each of the 9 generators: the expert helps where the VLM is weak")
    for i in range(9): ax.text(i + .2, orr[i] + .003, f"{(orr[i]-vlm[i])*100:+.1f}", ha="center", fontsize=8)
    save(fig, out, "20_per_generator_vlm_vs_or.png", "9 generators: VLM vs OR fusion")

    # heatmap of expert accuracy per generator / subset
    data = np.array([[.816, .870, .562, .645], [.915, .891, .511, .576], [.594, .644, .500, .522],
                     [.827, .837, .528, .569], [.766, .863, .550, .640], [.923, .932, .653, .811],
                     [.870, .889, .557, .658], [.960, .955, .645, .783], [.833, .867, .505, .505]])
    fig, ax = plt.subplots(figsize=(7.5, 5)); im = ax.imshow(data, cmap="YlGn", vmin=.45, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(["pixel\nofficial", "pixel\nours", "semantic\nofficial", "semantic\nours"])
    ax.set_yticks(range(9)); ax.set_yticklabels(GEN)
    for i in range(9):
        for j in range(4): ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Pixel expert accuracy per generator and subset (threshold 0.5)")
    fig.colorbar(im, ax=ax, shrink=.8)
    save(fig, out, "21_expert_heatmap.png", "Pixel expert accuracy heatmap per generator")

    # AUC per generator
    auc_p_off = [.964, .990, .818, .942, .969, .991, .985, .996, .976]
    auc_p_our = [.958, .967, .807, .926, .962, .984, .971, .993, .959]
    auc_s_off = [.735, .585, .552, .564, .726, .853, .743, .838, .526]
    auc_s_our = [.719, .647, .580, .601, .720, .888, .742, .863, .518]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True); x = np.arange(9)
    for a, (o, u, t) in zip(ax, [(auc_p_off, auc_p_our, "pixel subset"), (auc_s_off, auc_s_our, "semantic subset")]):
        a.bar(x - .2, o, .4, label="official", color="#31a354"); a.bar(x + .2, u, .4, label="ours", color="#9ecae1")
        a.set_xticks(x); a.set_xticklabels(GEN, rotation=40, ha="right"); a.set_title(f"AUC, {t}")
        a.axhline(.5, color="grey", ls=":"); a.set_ylim(.45, 1.0)
    ax[0].legend(); ax[0].set_ylabel("AUC (0.5 = random guessing)")
    save(fig, out, "22_expert_auc.png", "AUC of both pixel experts per generator")

    # complementarity
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    for a, (t, extra, fa) in zip(ax, [("pixel subset", 567, 78), ("semantic subset", 27, 269)]):
        a.bar(["extra fakes\nfound", "new false\nalarms"], [extra, fa], color=["#31a354", "#de2d26"])
        a.set_title(f"Official expert added to VLM: {t}\nnet {extra - fa:+d}")
        for i, v in enumerate([extra, fa]): a.text(i, v + 8, str(v), ha="center")
    save(fig, out, "23_complementarity.png", "Expert gain vs false alarms by subset")

    # confusion matrix of VLM
    cm = np.array([[17008, 992], [1462, 16538]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred real", "pred fake"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true real", "true fake"])
    for i in range(2):
        for j in range(2): ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12,
                                   color="white" if cm[i, j] > 8000 else "black")
    ax.set_title("VLM alone, all 36,000 images")
    save(fig, out, "24_vlm_confusion.png", "VLM confusion matrix (TN, FP, FN, TP)")

    # threshold sweep (our expert, full data)
    th = [.5, .8, .9, .95, .98]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(th, [.894, .922, .927, .932, .936], "o-", label="accuracy")
    ax.plot(th, [.828, .893, .908, .921, .937], "s--", label="real accuracy")
    ax.plot(th, [.961, .951, .946, .943, .935], "^--", label="fake accuracy")
    ax.axhline(.932, color="#e6550d", ls=":", label="VLM alone (.932)")
    ax.set_xlabel("expert threshold"); ax.set_ylabel("score"); ax.legend(fontsize=8)
    ax.set_title("VLM OR our expert: effect of the threshold (illustration, all data)")
    save(fig, out, "25_threshold_sweep.png", "Threshold sweep for our expert")

    # fusion methods test split
    names = ["VLM\nalone", "OR @0.5", "OR tuned\n(0.7)", "Agreement\ngate", "Learned\ngate"]
    acc = [.933, .939, .942, .941, .942]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(names, acc, color=["#e6550d", "#9ecae1", "#3182bd", "#74c476", "#31a354"])
    ax.set_ylim(.92, .95); ax.set_ylabel("accuracy, test split (25,200 images)")
    for i, v in enumerate(acc): ax.text(i, v + .0006, f"{v:.3f}", ha="center")
    ax.set_title("Gated fusion does not beat the tuned OR rule")
    save(fig, out, "26_fusion_methods.png", "Fusion methods on the test split")

    # cascade trade-off
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot([75.7, 72.2, 50.9, 47.4], [.936, .931, .921, .916], "o-", label="official expert")
    ax.plot([80.7, 73.9, 70.4, 62.4], [.936, .932, .927, .914], "s-", label="our expert")
    ax.scatter([100], [.933], color="red", zorder=5, label="VLM always (100 % calls)")
    ax.set_xlabel("share of images that still need the VLM (%)"); ax.set_ylabel("test accuracy")
    ax.invert_xaxis(); ax.legend(); ax.set_title("Cascade: accuracy vs. VLM calls")
    save(fig, out, "27_cascade_tradeoff.png", "Cascade accuracy vs. VLM calls")

    # audit
    cats = ["anatomy", "text", "lighting", "texture", "geometry", "context"]
    TP = [11.3, 19.3, 42.4, 46.2, 31.2, 25.1]; FN = [9.6, 15.7, 26.9, 24.4, 6.0, 22.3]
    FP = [11.8, 23.7, 30.6, 34.6, 13.2, 19.8]; TN = [11.0, 14.8, 12.9, 20.1, 3.4, 34.2]
    fig, ax = plt.subplots(figsize=(11, 4.4)); x = np.arange(6)
    for i, (v, l, c) in enumerate([(TP, "TP fake caught", "#d94801"), (FN, "FN fake missed", "#fdae6b"),
                                   (FP, "FP real called fake", "#6baed6"), (TN, "TN real called real", "#08519c")]):
        ax.bar(x + (i - 1.5) * .2, v, .2, label=l, color=c)
    ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("% of rationales naming this flaw")
    ax.legend(fontsize=8); ax.set_title("E4 audit: which flaw does the VLM name? (keyword-based, noisy)")
    save(fig, out, "28_audit.png", "E4 explanation audit")

    # training loss (printed points)
    steps = [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000]
    loss = [.715, .688, .616, .692, .431, .335, .286, .124, .460, .170, .446]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(steps, loss, "o-"); ax.set_xlabel("loader step"); ax.set_ylabel("loss of that single step")
    ax.set_title("Branch 2 training loss (11 logged single-step values, noisy)")
    save(fig, out, "29_training_loss.png", "Training loss at logged steps")


# ------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dda_real"); ap.add_argument("--dda_fake"); ap.add_argument("--aigi")
    ap.add_argument("--out", default="report_figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if a.dda_real and a.dda_fake:
        run("DDA pairs + spectrum", sec_dda_pairs, a.dda_real, a.dda_fake, a.out)
        run("Augmentation illustrations", sec_aug, a.dda_real, a.dda_fake, a.out)
    else:
        print("skip DDA sections (give --dda_real and --dda_fake)")
    if a.aigi: run("AIGI-Now overview", sec_aigi, a.aigi, a.out)
    else: print("skip AIGI-Now (give --aigi)")
    run("Diagrams", sec_diagrams, a.out)
    run("Result charts", sec_charts, a.out)

    with open(os.path.join(a.out, "MANIFEST.txt"), "w") as f:
        for n, d in MANIFEST: f.write(f"{n:45s} {d}\n")
    zp = shutil.make_archive(a.out, "zip", a.out)
    print(f"\nDONE. {len(MANIFEST)} items. Zip: {zp}")

if __name__ == "__main__":
    main()