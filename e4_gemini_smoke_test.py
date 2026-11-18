import os
import base64
import csv
import time
import argparse
import mimetypes
from pathlib import Path

from google import genai

BASE = Path(__file__).resolve().parent
INPUT_CSV = BASE / "eval/audit_sheet.csv"
IMAGE_DIR = BASE / "eval/audit_images"
OUTPUT_CSV = BASE / "eval/gemini_rationales_100.csv"

MODEL = "gemini-3.8-flash"

CUES = {
    "anatomy": ["hand", "finger", "eye", "face", "teeth", "limb",
                "anatomy", "body", "ear", "arm", "leg"],
    "text": ["text", "letter", "writing", "sign", "logo", "label"],
    "lighting": ["lighting", "shadow", "reflection", "light "],
    "texture": ["texture", "smooth", "plastic", "skin", "blur",
                "grain", "noise", "sharp", "detail"],
    "geometry": ["perspective", "geometry", "symmetr", "proportion",
                 "physics", "gravity", "distort"],
    "context": ["background", "context", "scene", "object", "composition"],
}

PROMPT = """
Carefully inspect this image for visible signs of image manipulation
or AI generation.

Consider these categories:
1. Anatomy: unusual body parts, faces, hands or proportions.
2. Text: distorted or inconsistent letters, logos or signs.
3. Lighting: inconsistent illumination, shadows or reflections.
4. Texture: unnatural skin, surfaces, blur, repeated patterns or details.
5. Geometry: inconsistent perspective, shapes or spatial relationships.
6. Context: objects or scene elements that do not fit together.

Important instructions:
- Describe only specific visual evidence actually visible in the image.
- Do not assume the image is fake.
- Do not invent defects just to fill the categories.
- If you cannot identify a clear visual inconsistency, say so.
- Do not make a final real/fake classification.
- Give a concise explanation in approximately 80-120 words.
"""


def extract_cues(text):
    text = (text or "").lower()
    found = []

    for category, keywords in CUES.items():
        if any(word in text for word in keywords):
            found.append(category)

    return found


def load_existing_successes():
    successful = {}

    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "SUCCESS":
                    successful[row["id"]] = row

    return successful


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is missing.")

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    client = genai.Client(api_key=api_key)

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = rows[:args.limit]

    successful = load_existing_successes()

    fields = [
        "id", "outcome", "generator", "subset", "image_file",
        "original_vlm_rationale", "gemini_rationale",
        "vlm_cues", "gemini_cues",
        "common_cues", "only_vlm", "only_gemini",
        "status", "error"
    ]

    print("Model:", MODEL)
    print("Images requested:", len(rows))
    print("Previously successful:", len(successful))
    print("Images to process:", sum(1 for r in rows if r["id"] not in successful))

    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if write_header:
            writer.writeheader()

        for index, row in enumerate(rows, 1):
            image_id = row["id"]

            if image_id in successful:
                print(f"[{index}/{len(rows)}] ID {image_id}: already completed, skipping")
                continue

            image_path = IMAGE_DIR / row["image_file"]

            if not image_path.is_file():
                print(f"[{index}/{len(rows)}] Missing image: {image_path}")
                continue

            print(f"\n[{index}/{len(rows)}] Processing {row['image_file']} ...")

            result = {
                "id": image_id,
                "outcome": row["outcome"],
                "generator": row["generator"],
                "subset": row["subset"],
                "image_file": row["image_file"],
                "original_vlm_rationale": row["rationale"],
                "gemini_rationale": "",
                "vlm_cues": row["auto_cues"],
                "gemini_cues": "",
                "common_cues": "",
                "only_vlm": "",
                "only_gemini": "",
                "status": "ERROR",
                "error": ""
            }

            try:
                mime_type, _ = mimetypes.guess_type(str(image_path))
                mime_type = mime_type or "image/jpeg"

                image_bytes = image_path.read_bytes()

                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                response = client.interactions.create(
                    model=MODEL,
                    input=[
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image",
                            "data": image_b64,
                            "mime_type": mime_type
                        }
                    ]
                )

                gemini_text = (response.output_text or "").strip()

                if not gemini_text:
                    raise RuntimeError("Gemini returned an empty response.")

                vlm_cues = set(
                    c.strip() for c in row["auto_cues"].split(",") if c.strip()
                )
                gemini_cues = set(extract_cues(gemini_text))

                result.update({
                    "gemini_rationale": gemini_text,
                    "gemini_cues": ",".join(sorted(gemini_cues)),
                    "common_cues": ",".join(sorted(vlm_cues & gemini_cues)),
                    "only_vlm": ",".join(sorted(vlm_cues - gemini_cues)),
                    "only_gemini": ",".join(sorted(gemini_cues - vlm_cues)),
                    "status": "SUCCESS",
                    "error": ""
                })

                print("Gemini:", gemini_text[:250].replace("\n", " "))
                print("VLM cues:", sorted(vlm_cues))
                print("Gemini cues:", sorted(gemini_cues))
                print("Common:", sorted(vlm_cues & gemini_cues))

            except Exception as e:
                result["error"] = str(e)[:1000]
                print("ERROR:", result["error"])

            writer.writerow(result)
            f.flush()

            time.sleep(1)

    print("\n================================")
    print("Processing finished.")
    print("Output:", OUTPUT_CSV)
    print("================================")


if __name__ == "__main__":
    main()
