import os
import base64
import csv
import time
import argparse
import mimetypes
from pathlib import Path

from openai import OpenAI

BASE = Path(__file__).resolve().parent
INPUT_CSV = BASE / "eval/audit_sheet.csv"
IMAGE_DIR = BASE / "eval/audit_images"
OUTPUT_CSV = BASE / "eval/openrouter_rationales_100.csv"

MODEL = "nvidia/nemotron-3.5-super-vl:free"

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
    return [
        category for category, keywords in CUES.items()
        if any(word in text for word in keywords)
    ]


def load_existing_successes():
    successful = {}

    if OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "SUCCESS":
                    successful[row["id"]] = row

    return successful


def get_response_text(response):
    content = response.choices[0].message.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()

    return str(content).strip() if content else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY environment variable is missing."
        )

    if not INPUT_CSV.exists():
        raise FileNotFoundError(INPUT_CSV)

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = rows[:args.limit]
    successful = load_existing_successes()

    fields = [
        "id", "outcome", "generator", "subset", "image_file",
        "original_vlm_rationale", "openrouter_rationale",
        "vlm_cues", "openrouter_cues",
        "common_cues", "only_vlm", "only_openrouter",
        "status", "error"
    ]

    print("Model:", MODEL)
    print("Images requested:", len(rows))
    print("Previously successful:", len(successful))
    print(
        "Images remaining:",
        sum(1 for r in rows if r["id"] not in successful)
    )

    write_header = (
        not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
    )

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        if write_header:
            writer.writeheader()

        for index, row in enumerate(rows, 1):
            image_id = row["id"]

            if image_id in successful:
                print(
                    f"[{index}/{len(rows)}] ID {image_id}: "
                    "already completed, skipping"
                )
                continue

            image_path = IMAGE_DIR / row["image_file"]

            if not image_path.is_file():
                print(f"Missing image: {image_path}")
                continue

            print(
                f"\n[{index}/{len(rows)}] "
                f"Processing {row['image_file']} ..."
            )

            result = {
                "id": image_id,
                "outcome": row["outcome"],
                "generator": row["generator"],
                "subset": row["subset"],
                "image_file": row["image_file"],
                "original_vlm_rationale": row["rationale"],
                "openrouter_rationale": "",
                "vlm_cues": row["auto_cues"],
                "openrouter_cues": "",
                "common_cues": "",
                "only_vlm": "",
                "only_openrouter": "",
                "status": "ERROR",
                "error": ""
            }

            try:
                mime_type, _ = mimetypes.guess_type(str(image_path))
                mime_type = mime_type or "image/jpeg"

                image_bytes = image_path.read_bytes()
                image_b64 = base64.b64encode(
                    image_bytes
                ).decode("utf-8")

                response = None

                for attempt in range(3):
                    try:
                        response = client.chat.completions.create(
                            model=MODEL,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": PROMPT
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": (
                                                    f"data:{mime_type};"
                                                    f"base64,{image_b64}"
                                                )
                                            }
                                        }
                                    ]
                                }
                            ]
                        )
                        break

                    except Exception as api_error:
                        status_code = getattr(
                            api_error, "status_code", None
                        )

                        if status_code == 429:
                            raise

                        if status_code in (500, 502, 503, 504):
                            if attempt < 2:
                                wait = 5 * (2 ** attempt)
                                print(
                                    f"Temporary server error. "
                                    f"Retrying in {wait}s..."
                                )
                                time.sleep(wait)
                                continue

                        raise

                openrouter_text = get_response_text(response)

                if not openrouter_text:
                    raise RuntimeError(
                        "OpenRouter returned an empty response."
                    )

                vlm_cues = set(
                    c.strip()
                    for c in row["auto_cues"].split(",")
                    if c.strip()
                )

                openrouter_cues = set(
                    extract_cues(openrouter_text)
                )

                result.update({
                    "openrouter_rationale": openrouter_text,
                    "openrouter_cues": ",".join(
                        sorted(openrouter_cues)
                    ),
                    "common_cues": ",".join(
                        sorted(vlm_cues & openrouter_cues)
                    ),
                    "only_vlm": ",".join(
                        sorted(vlm_cues - openrouter_cues)
                    ),
                    "only_openrouter": ",".join(
                        sorted(openrouter_cues - vlm_cues)
                    ),
                    "status": "SUCCESS",
                    "error": ""
                })

                print(
                    "OpenRouter:",
                    openrouter_text[:250].replace("\n", " ")
                )
                print("VLM cues:", sorted(vlm_cues))
                print("OpenRouter cues:", sorted(openrouter_cues))
                print("Common cues:", sorted(vlm_cues & openrouter_cues))

            except Exception as e:
                result["error"] = str(e)[:1000]
                print("ERROR:", result["error"])

            writer.writerow(result)
            f.flush()

            if "status_code" in locals() and False:
                pass

            # A 429 may indicate exhausted daily quota.
            # Stop instead of repeatedly sending requests.
            if result["status"] == "ERROR" and "429" in result["error"]:
                print(
                    "\nRate limit/quota error detected. "
                    "Stopping safely. Run again later to resume."
                )
                break

            # Keep request frequency conservative.
            time.sleep(3)

    print("\n================================")
    print("Processing finished.")
    print("Output:", OUTPUT_CSV)
    print("================================")


if __name__ == "__main__":
    main()
