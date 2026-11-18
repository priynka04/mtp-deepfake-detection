import json, sys
from pathlib import Path
sys.path.insert(0, "/DATA/storage/ab_students/align_gemini_project/Align-Dont-Blend")
from detector.core import VLM_SYSTEM_PROMPT, VLM_USER_PROMPT, read_inputs

recs = read_inputs(Path(sys.argv[1]).resolve())
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for r in recs:
        msgs = [{"role": "system", "content": VLM_SYSTEM_PROMPT},
                {"role": "user", "content": VLM_USER_PROMPT}]
        f.write(json.dumps({"messages": msgs, "images": [r["image"]]}, ensure_ascii=False) + "\n")
print(len(recs), "rows")
