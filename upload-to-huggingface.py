"""
Upload Naat Training Data to HuggingFace

Reads the static training-data/ folder committed in this repo
and uploads it as a HuggingFace dataset using the Hub API directly.
No torchcodec/soundfile dependency needed.

Usage:
  1. pip install -r requirements.txt
  2. huggingface-cli login  (paste your HF token)
  3. python upload-to-huggingface.py

The dataset will be available at:
  https://huggingface.co/datasets/<your-username>/naat-classifier
"""

import json
import os
import csv
import io
from huggingface_hub import HfApi, create_repo

# ── Config ────────────────────────────────────────────────────
TRAINING_DATA_DIR = os.path.join(os.getcwd(), "training-data")
MANIFEST_PATH = os.path.join(TRAINING_DATA_DIR, "manifest.json")
HF_DATASET_NAME = "naat-classifier"


def clear_existing_dataset(api: HfApi, repo_id: str):
    """
    Remove previously uploaded dataset files so each run replaces the repo contents.
    """
    print("\n🧹 Clearing previous dataset files from HuggingFace...")
    try:
        api.delete_files(
            repo_id=repo_id,
            repo_type="dataset",
            delete_patterns=[
                "metadata.csv",
                "naat/",
                "explanation/",
                "data/",
                "data/naat/",
                "data/explanation/",
            ],
            commit_message="Clear previous training dataset files",
        )
        print("  ✅ Previous dataset files deleted")
    except Exception as error:
        print(f"  ℹ️  Nothing to clear or delete skipped: {error}")

# ── Load manifest ─────────────────────────────────────────────
with open(MANIFEST_PATH, "r") as f:
    manifest = json.load(f)

print(f"📋 Loaded {len(manifest)} chunks from manifest")

# ── Validate files exist ──────────────────────────────────────
valid = []
for entry in manifest:
    filepath = os.path.join(TRAINING_DATA_DIR, entry["file"])
    if os.path.exists(filepath):
        valid.append(entry)
    else:
        print(f"  ⚠️  Missing: {entry['file']}, skipping")

naat_count = sum(1 for e in valid if e["label"] == "naat")
expl_count = sum(1 for e in valid if e["label"] == "explanation")
print(f"✅ Found {len(valid)} valid audio files")
print(f"   Naat: {naat_count}, Explanation: {expl_count}")

# ── Create HuggingFace repo ──────────────────────────────────
api = HfApi()
user = api.whoami()["name"]
repo_id = f"{user}/{HF_DATASET_NAME}"

print(f"\n📦 Creating dataset repo: {repo_id}")
create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
clear_existing_dataset(api, repo_id)

# ── Build metadata CSV ────────────────────────────────────────
csv_buffer = io.StringIO()
writer = csv.writer(csv_buffer)
writer.writerow(["file_name", "label", "source", "start", "end"])
for entry in valid:
    writer.writerow([
        entry["file"],
        entry["label"],
        entry["source"],
        entry["start"],
        entry["end"],
    ])

# ── Upload files ──────────────────────────────────────────────
print(f"\n🚀 Uploading {len(valid)} audio files...")

# Upload metadata.csv first.
api.upload_file(
    path_or_fileobj=csv_buffer.getvalue().encode("utf-8"),
    path_in_repo="metadata.csv",
    repo_id=repo_id,
    repo_type="dataset",
    commit_message="Add metadata.csv",
)
print("  ✅ metadata.csv uploaded")

# Upload all WAVs via resumable large-folder upload.
# This is more robust for thousands of files and can resume progress.
api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=TRAINING_DATA_DIR,
    allow_patterns=["naat/*.wav", "explanation/*.wav"],
)
print(f"  ✅ {len(valid)} audio files uploaded")

print(f"\n✅ Done! Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")
print(f"   Use in Colab: load_dataset('{repo_id}')")
