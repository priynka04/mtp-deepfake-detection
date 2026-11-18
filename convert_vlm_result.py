import json, sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src, encoding="utf-8") as f, open(dst, "x", encoding="utf-8") as g:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        imgs = row["images"]
        assert len(imgs) == 1, imgs
        v = imgs[0]
        if isinstance(v, dict):
            v = v.get("path")
        assert isinstance(v, str), repr(imgs[0])[:200]
        g.write(json.dumps({"image": v, "response": row["response"]}, ensure_ascii=False) + "\n")
        n += 1
print(n, "rows converted")
