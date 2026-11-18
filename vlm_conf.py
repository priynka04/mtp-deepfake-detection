import json, math, sys

def _tokens(lp):
    if lp is None:
        return None
    if isinstance(lp, dict) and "content" in lp:
        return lp["content"]
    if isinstance(lp, list):
        return lp
    return None

def _p_tagged(tokens):
    if not tokens:
        return None
    cum, seen = "", False
    for t in tokens:
        tok = t.get("token", "")
        cum += tok
        if not seen:
            seen = "<answer>" in cum
            continue
        s = tok.strip().lower()
        if not (s.startswith("auth") or s.startswith("ai")):
            continue
        pf = pa = 0.0
        cands = t.get("top_logprobs") or [{"token": tok, "logprob": t["logprob"]}]
        for x in cands:
            c = x["token"].strip().lower()
            if c.startswith("ai"):
                pf += math.exp(x["logprob"])
            elif c.startswith("auth"):
                pa += math.exp(x["logprob"])
        return pf / (pf + pa) if pf + pa > 0 else None
    return None

def _p_untagged(tokens):
    """Jawab bina <answer> tag ke ho: '... This is an AI-generated image.' (aakhri vakya)"""
    if not tokens:
        return None
    cum, result = "", None
    for t in tokens:
        tok = t.get("token", "")
        s_ = tok.strip().lower()
        if (s_.startswith("auth") or s_.startswith("ai")) and cum.rstrip().lower().endswith("this is an"):
            pf = pa = 0.0
            cands = t.get("top_logprobs") or [{"token": tok, "logprob": t["logprob"]}]
            for x in cands:
                c = x["token"].strip().lower()
                if c.startswith("ai"):
                    pf += math.exp(x["logprob"])
                elif c.startswith("auth"):
                    pa += math.exp(x["logprob"])
            result = pf / (pf + pa) if pf + pa > 0 else None
        cum += tok
    return result

def p_fake_from_tokens(tokens):
    p = _p_tagged(tokens)
    return p if p is not None else _p_untagged(tokens)

def p_fake_from_row(row):
    return p_fake_from_tokens(_tokens(row.get("logprobs")))

if __name__ == "__main__":
    import glob, os
    sys.path.insert(0, "/DATA/storage/ab_students/align_gemini_project/Align-Dont-Blend")
    from detector.core import parse_vlm_verdict
    d = sys.argv[1]
    out, n, ok, agree, shown = [], 0, 0, 0, 0
    for f in sorted(glob.glob(os.path.join(d, "*.result.jsonl"))):
        for l in open(f):
            r = json.loads(l); n += 1
            v = r["images"][0]; path = v["path"] if isinstance(v, dict) else v
            p = p_fake_from_row(r)
            try: pv = bool(parse_vlm_verdict(r["response"]))
            except Exception: pv = None
            if p is None:
                if shown < 2:
                    print("NO p_fake for:", path, "| logprobs type:", type(r.get("logprobs")).__name__,
                          "| sample:", str(r.get("logprobs"))[:300]); shown += 1
                continue
            ok += 1; agree += int(pv is not None and (p > 0.5) == pv)
            out.append({"image": path, "p_fake": p, "verdict": pv})
    with open(os.path.join(d, "conf.jsonl"), "w") as g:
        for o in out: g.write(json.dumps(o) + "\n")
    print(f"rows={n} | p_fake found={ok} ({ok/max(n,1):.1%}) | agrees with parse_vlm_verdict={agree/max(ok,1):.1%}")
    if out:
        ps = [o["p_fake"] for o in out]
        print(f"p_fake>0.99 or <0.01 (saturated): {sum(1 for p in ps if p>0.99 or p<0.01)/len(ps):.1%}")
    print("saved:", os.path.join(d, "conf.jsonl"))
