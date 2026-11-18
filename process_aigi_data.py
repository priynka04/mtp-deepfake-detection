import os
import tarfile
import json
import glob

def extract_and_format():
    tar_path = "data/semantic_preference/AIGI-Now.tar.gz"
    extract_dir = "data/semantic_preference/extracted"
    output_jsonl = "data/vlm_dpo_train.jsonl"
    
    if os.path.exists(tar_path):
        print(f"Extracting {tar_path} into {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("Extraction complete!")
    else:
        print(f"Archive file not found yet at {tar_path}!")
        return

    # Generate JSONL for ms-swift DPO
    print(f"Formatting dataset into {output_jsonl}...")
    image_files = glob.glob(os.path.join(extract_dir, "**/*.jpg"), recursive=True) + \
                  glob.glob(os.path.join(extract_dir, "**/*.png"), recursive=True)
    
    dpo_records = []
    for img_path in image_files[:5000]:  # Initial training subset limit
        is_fake = "fake" in img_path.lower() or "aigi" in img_path.lower()
        
        record = {
            "query": "Analyze this image for AI generation artifacts, lighting consistency, and anatomical structures. Is this real or a deepfake?",
            "response_chosen": "The image exhibits unnatural lighting blending and synthetic textures. Therefore, it is identified as AI-generated/deepfake." if is_fake else "The lighting, texture, and pixel structures are consistent with natural camera captures. Therefore, it is real.",
            "response_rejected": "This image looks completely natural and authentic." if is_fake else "This image is clearly AI-generated with severe artifacts.",
            "images": [img_path]
        }
        dpo_records.append(record)
        
    with open(output_jsonl, "w") as f:
        for entry in dpo_records:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully generated {len(dpo_records)} DPO samples in {output_jsonl}!")

if __name__ == "__main__":
    extract_and_format()
