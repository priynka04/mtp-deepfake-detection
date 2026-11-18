import random, re
import gradio as gr
from common import *

recs = load_all()
print("loaded", len(recs))

def parts(resp):
    resp = resp or ""
    m = re.search(r"<rationale>(.*?)</rationale>", resp, re.S | re.I)
    a = re.search(r"<answer>(.*?)</answer>", resp, re.S | re.I)
    rat = " ".join(m.group(1).split()) if m else " ".join(resp.split())
    ans = " ".join(a.group(1).split()) if a else "(no <answer> tag)"
    return rat, ans

# presentation ke liye ready-made stories
SCEN = {
 "Any image": lambda r, t: True,
 "VLM missed the fake, expert caught it": lambda r, t: r["y"] == 1 and not r["v"] and r["o"] >= t,
 "VLM caught the fake, expert missed it": lambda r, t: r["y"] == 1 and r["v"] and r["o"] < t,
 "Both caught the fake": lambda r, t: r["y"] == 1 and r["v"] and r["o"] >= t,
 "Both missed the fake": lambda r, t: r["y"] == 1 and not r["v"] and r["o"] < t,
 "Real image: expert false alarm": lambda r, t: r["y"] == 0 and not r["v"] and r["o"] >= t,
 "Real image: VLM false alarm": lambda r, t: r["y"] == 0 and r["v"],
 "Real image: both correct": lambda r, t: r["y"] == 0 and not r["v"] and r["o"] < t,
}

def pick(gen, sub, label, scen, thr, reveal):
    f = SCEN[scen]
    pool = [r for r in recs
            if (gen == "All" or r["gen"] == gen)
            and (sub == "All" or r["sub"] == sub)
            and (label == "Any" or r["y"] == int(label == "fake"))
            and f(r, thr)]
    if not pool:
        return None, "No image matches these filters.", "", {}, {}
    r = random.choice(pool)
    rat, ans = parts(r["resp"])
    final = bool(r["v"]) or r["o"] >= thr
    md = (f"**Pool size:** {len(pool)} images match | **Generator:** {r['gen']} | **Subset:** {r['sub']}\n\n"
          f"**Semantic branch (VLM) verdict:** {'AI-generated' if r['v'] else 'Authentic'}\n\n"
          f"**Pixel expert (official @ {thr:.2f}):** {'fake' if r['o'] >= thr else 'real'}\n\n"
          f"### Final (OR rule): {'FAKE' if final else 'REAL'}")
    if reveal:
        truth = "FAKE" if r["y"] else "REAL"
        ok = final == bool(r["y"])
        md += f"\n\n**Ground truth:** {truth}  {'✅ correct' if ok else '❌ wrong'}"
    return (r["path"], md, rat,
            {"fake": float(r["o"]), "real": float(1 - r["o"])},
            {"fake": float(r["u"]), "real": float(1 - r["u"])})

SUMMARY = """
### Main numbers (AIGI-Now, 36,000 images, threshold 0.5)
| System | Accuracy | Real recall | Fake recall |
|---|---|---|---|
| VLM alone | 0.932 | 0.945 | 0.919 |
| Pixel expert (official) | 0.695 | 0.979 | 0.412 |
| Pixel expert (ours) | 0.748 | 0.870 | 0.625 |
| VLM OR official | 0.939 | 0.926 | 0.952 |
| VLM OR ours | 0.894 | 0.828 | 0.961 |

On the held-out 70% test split, VLM OR official with a tuned threshold (0.7) reaches 0.942 (VLM alone: 0.933).
"""

with gr.Blocks(title="Two-branch deepfake detector") as demo:
    gr.Markdown("## Two-branch AI-image detector: semantic branch (Qwen2.5-VL) + pixel expert")
    with gr.Tab("Examples"):
        with gr.Row():
            gen = gr.Dropdown(["All"] + GENS, value="All", label="Generator")
            sub = gr.Dropdown(["All", "pixel", "semantic"], value="All", label="Subset")
            label = gr.Dropdown(["Any", "real", "fake"], value="Any", label="Ground truth")
        scen = gr.Dropdown(list(SCEN), value="Any image", label="Story / scenario")
        thr = gr.Slider(0.1, 0.99, value=0.7, step=0.01, label="Expert threshold (official)")
        reveal = gr.Checkbox(True, label="Show ground truth")
        btn = gr.Button("Show random image", variant="primary")
        with gr.Row():
            img = gr.Image(type="filepath", label="Image", height=420)
            with gr.Column():
                info = gr.Markdown()
                rat = gr.Textbox(label="Semantic branch explanation (Qwen2.5-VL)", lines=8)
        with gr.Row():
            lo = gr.Label(label="Pixel expert: official (fake probability)")
            lu = gr.Label(label="Pixel expert: ours (fake probability)")
        btn.click(pick, [gen, sub, label, scen, thr, reveal], [img, info, rat, lo, lu])
    with gr.Tab("Summary numbers"):
        gr.Markdown(SUMMARY)

demo.launch(server_name="127.0.0.1", server_port=7860, allowed_paths=[str(P)])
