import json
import os
import glob
from pathlib import Path

def prepare_swift_dpo_data(data_dir, output_file):
    print(f"Processing preference dataset from: {data_dir}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check downloaded files
    tar_files = glob.glob(os.path.join(data_dir, "*.tar.gz")) + glob.glob(os.path.join(data_dir, "*.zip"))
    print(f"Found compressed archives: {tar_files}")
    
    print("Dataset format script ready. Will execute formatting once extraction finishes.")

if __name__ == "__main__":
    prepare_swift_dpo_data("data/semantic_preference", "data/vlm_dpo_train.jsonl")
