import base64, json, math, os, re, sys
from pathlib import Path
import gradio as gr
import httpx

P = Path("/DATA/storage/ab_students/align_gemini_project")
sys.path.insert(0, str(P / "Align-Dont-Blend"))
URL = "http://127.0.0.1:8000"

try:
    from detector.core import parse_vlm_verdict
except Exception:
    parse_vlm_verdict = None

# training wala same prompt, kisi purane result se uthaya
first = json.loads(open(P / "eval/vlm_full/Nano.result.jsonl").readline())
SYSTEM = first["messages"][0]["content"]
USER_TEXT = first["messages"][1]["content"].replace("<image>", "").strip()
print("SYSTEM PROMPT:", SYSTEM[:300].replace("\n", " "), "...")
print("USER TEXT:", repr(USER_TEXT[:300]))

def get_model():
    ids = [m["id"] for m in httpx.get(URL + "/v1/models", timeout=10).json()["data"]]
    print("server models:", ids)
    if os.environ.get("VLM_MODEL"):
        return os.environ["VLM_MODEL"]
    for i in ids:
        if "adapter" in i.lower():
            return i
    return ids[0]

def verdict(resp):
    if parse_vlm_verdict:
        try:
            return bool(parse_vlm_verdict(resp))
        except Exception:
            return None
    m = re.search(r"<answer>(.*?)</answer>", resp, re.S | re.I)
    a = (m.group(1) if m else "").lower()
    return None if not a else ("authentic" not in a)

def parts(resp):
    m = re.search(r"<rationale>(.*?)</rationale>", resp or "", re.S | re.I)
    return " ".join(m.group(1).split()) if m else " ".join((resp or "").split())

def confidence(choice):
    """<answer> ke baad pehla token jahan 'AI' ya 'authentic' ka mukabla hota hai."""
    try:
        toks = choice["logprobs"]["content"]
    except Exception:
        return None
    cum, seen = "", False
    for t in toks:
        cum += t["token"]
        if not seen:
            seen = "<answer>" in cum
            continue
        pf = pr = 0.0
        for x in t.get("top_logprobs", []):
            s = x["token"].strip().lower()
            if s.startswith("ai"):
                pf += math.exp(x["logprob"])
            elif s.startswith("auth"):
                pr += math.exp(x["logprob"])
        if pf + pr > 0:
            return pf / (pf + pr)
    return None

def call_vlm(path):
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    content = [{"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]
    if USER_TEXT:
        content.append({"type": "text", "text": USER_TEXT})
    body = {"model": get_model(), "temperature": 0, "max_tokens": 512,
            "logprobs": True, "top_logprobs": 5,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": content}]}
    r = httpx.post(URL + "/v1/chat/completions", json=body, timeout=300)
    r.raise_for_status()
    ch = r.json()["choices"][0]
    return ch["message"]["content"], ch

def analyze(path):
    if not path:
        return "Pehle image upload karo.", {}, ""
    try:
        resp, choice = call_vlm(path)
    except Exception as e:
        return f"VLM server error: {e}\n\nCheck: logs/vlm_server.log", {}, ""
    v = verdict(resp)
    pf = confidence(choice)
    if v is None:
        head = "### Result: UNCLEAR (model ka final verdict parse nahi hua)"
    else:
        head = f"### Result: {'AI-GENERATED (FAKE)' if v else 'AUTHENTIC (REAL)'}"
    if pf is None:
        head += "\n\nVLM confidence: n/a (server ne logprobs nahi diye)"
        label = {}
    else:
        head += f"\n\nVLM confidence: **{max(pf, 1 - pf):.1%}**"
        label = {"AI-generated": float(pf), "Authentic": float(1 - pf)}
    return head, label, parts(resp)

with gr.Blocks(title="Two-branch deepfake detector") as demo:
    gr.Markdown("## Upload an image: semantic branch (Qwen2.5-VL) verdict + explanation")
    with gr.Row():
        img = gr.Image(type="filepath", label="Upload image", height=420)
        with gr.Column():
            out = gr.Markdown()
            lab = gr.Label(label="VLM confidence")
            rat = gr.Textbox(label="Explanation (semantic branch)", lines=8)
    gr.Button("Analyze", variant="primary").click(analyze, img, [out, lab, rat])

demo.launch(server_name="127.0.0.1", server_port=7860)
