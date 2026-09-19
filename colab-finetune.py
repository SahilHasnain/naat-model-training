"""
Naat vs Explanation Classifier — Fine-tune Wav2Vec2

Two modes:

A) GPU / Colab (default):
   Use a GPU runtime to fine-tune the full model. Loads the dataset from your
   HuggingFace dataset repo and pushes the model back to the Hub.

   Instructions:
     1. Open Google Colab (https://colab.research.google.com)
     2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
     3. Copy-paste this entire file into a single cell and run it
     4. When prompted, paste your HuggingFace token (needs write access)
     5. Training takes ~5-10 min on T4 with 44 samples
     6. Model is pushed to: huggingface.co/<your-user>/naat-classifier-model

B) Local CPU (no HuggingFace dataset needed):
   Runs anywhere with `python` + the requirements installed. Reads the static
   data straight from `training-data/`, so the HF dataset repo is NOT touched.

     python colab-finetune.py --local --freeze-base
     python colab-finetune.py --local --freeze-base --no-push

   Flags:
     --local            Load from ./training-data instead of the HF dataset repo
     --data-dir PATH    Where the static data lives (default: ./training-data)
     --freeze-base      Freeze wav2vec2 backbone, train only the head (fast CPU)
     --no-push          Save model locally, don't push to HF Hub
     --output-dir PATH  Where to save checkpoints (default: ./naat-classifier-checkpoints)
     --epochs N         Override num_train_epochs (default: 10)
     --batch-size N     Override per-device batch size (default: 8)
"""

import argparse
import json
import os
import subprocess
import sys

# Windows consoles default to cp1252, which can't print emoji/unicode. The
# script is designed for both Colab (UTF-8) and local CPU runs, so force it.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

def build_args():
    parser = argparse.ArgumentParser(description="Naat vs Explanation classifier fine-tune")
    parser.add_argument("--local", action="store_true",
                        help="Load from ./training-data instead of the HF dataset repo")
    parser.add_argument("--data-dir", default="training-data",
                        help="Path to static training data (default: ./training-data)")
    parser.add_argument("--freeze-base", action="store_true",
                        help="Freeze wav2vec2 backbone, train only the head (CPU-friendly)")
    parser.add_argument("--no-push", action="store_true",
                        help="Save model locally, don't push to HF Hub")
    parser.add_argument("--output-dir", default="./naat-classifier-checkpoints",
                        help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_train_epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size")
    return parser.parse_args()

ARGS = build_args()

# ── 1. Install dependencies ──────────────────────────────────
def pip_install(packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])

def ensure_deps():
    try:
        import datasets, transformers, accelerate, evaluate, soundfile
    except ImportError:
        print("📦 Installing dependencies...")
        pip_install(["datasets", "transformers", "accelerate", "evaluate",
                     "huggingface_hub", "soundfile", "librosa"])

ensure_deps()

from datasets import load_dataset, Dataset, Audio, ClassLabel
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
from huggingface_hub import login, HfApi
import numpy as np
import evaluate

MODEL_NAME = "facebook/wav2vec2-base"
label2id = {"naat": "0", "explanation": "1"}
id2label = {"0": "naat", "1": "explanation"}

# ── 2. Authentication (only needed when pushing to HF) ───────
def login_if_needed():
    from huggingface_hub import login
    if ARGS.no_push:
        return None
    token = os.environ.get("HF_TOKEN", "").strip()
    print("\n🔑 Logging in to HuggingFace...")
    if token:
        login(token)  # headless: HF_TOKEN env var, no prompt
        print("   Authenticated via HF_TOKEN")
    else:
        login()  # Will prompt for token interactively — never hardcode tokens!
    api = HfApi()
    username = api.whoami()["name"]
    print(f"   Logged in as: {username}")
    return username

username = login_if_needed()

# ── 3. Load dataset ──────────────────────────────────────────
print("\n📋 Loading dataset...")

def load_from_hf(user):
    ds = load_dataset(
        f"{user}/naat-classifier",
        revision="main",
        download_mode="force_redownload",
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds["train"].cast_column("label", ClassLabel(names=["naat", "explanation"]))

def load_from_local(data_dir):
    import soundfile as sf

    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit(f"❌ {manifest_path} not found. Run to-builder or check --data-dir.")
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Read the WAVs directly into float arrays so we don't depend on
    # datasets' audio decoder (torchcodec/soundfile) at map time.
    rows = []
    for entry in manifest:
        path = os.path.join(data_dir, entry["file"])
        if not os.path.exists(path):
            sys.exit(f"❌ Chunk missing: {path} (stale manifest — rebuild the data)")
        array, sr = sf.read(path, dtype="float32", always_2d=False, frames=16000 * 5)
        rows.append({"audio": {"array": array, "sampling_rate": int(sr)}, "label": entry["label"]})

    ds = Dataset.from_list(rows)
    return ds.cast_column("label", ClassLabel(names=["naat", "explanation"]))

if ARGS.local:
    dataset = load_from_local(ARGS.data_dir)
    print(f"   Total samples: {len(dataset)} (local: {ARGS.data_dir})")
else:
    if username is None:
        sys.exit("❌ --no-push requires --local (HF dataset mode always pushes).")
    dataset = load_from_hf(username)
    print(f"   Total samples: {len(dataset)}")

print(f"   Features: {dataset.features}")

# ── 4. Train/test split ──────────────────────────────────────
dataset = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
print(f"   Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

# ── 5. Load feature extractor & model ────────────────────────
OUTPUT_MODEL = f"{username or 'local'}/naat-classifier-model"

print(f"\n🧠 Loading {MODEL_NAME}...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    label2id=label2id,
    id2label=id2label,
)

if ARGS.freeze_base:
    # Freeze the wav2vec2 backbone; only the classification head stays trainable.
    # 95M fewer trainable params -> CPU-friendly without touching inference code.
    for param in model.wav2vec2.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   🔒 Backbone frozen. Trainable params: {trainable:,} (head only)")

# ── 6. Preprocess (batched, following official HF guide) ─────
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    return feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        max_length=16000 * 5,  # 5 seconds
        truncation=True,
    )

print("⚙️  Preprocessing audio...")
encoded = dataset.map(preprocess_function, remove_columns="audio", batched=True)

# ── 7. Training setup ────────────────────────────────────────
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir=ARGS.output_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5 if not ARGS.freeze_base else 1e-3,
    per_device_train_batch_size=ARGS.batch_size,
    per_device_eval_batch_size=ARGS.batch_size,
    num_train_epochs=ARGS.epochs if ARGS.epochs is not None else 10,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=(not ARGS.no_push),
    hub_model_id=None if ARGS.no_push else OUTPUT_MODEL,
    hub_private_repo=(not ARGS.no_push),
    save_only_model=ARGS.no_push,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded["train"],
    eval_dataset=encoded["test"],
    processing_class=feature_extractor,
    compute_metrics=compute_metrics,
)

# ── 8. Train ──────────────────────────────────────────────────
print("\n🚀 Starting training...")
trainer.train()

# ── 9. Evaluate ───────────────────────────────────────────────
print("\n📊 Final evaluation:")
results = trainer.evaluate()
print(f"   Accuracy: {results['eval_accuracy']:.1%}")
print(f"   Loss: {results['eval_loss']:.4f}")

# ── 10. Save / push ──────────────────────────────────────────
if ARGS.no_push:
    local_dir = os.path.abspath(os.path.join(ARGS.output_dir, "final-model"))
    model.save_pretrained(local_dir)
    feature_extractor.save_pretrained(local_dir)
    print(f"\n✅ Model saved locally: {local_dir}")
    print("   To use it: pipeline('audio-classification', model=<local_dir>)")
else:
    print(f"\n📤 Pushing model to {OUTPUT_MODEL}...")
    trainer.push_to_hub()
    feature_extractor.push_to_hub(OUTPUT_MODEL)

    print(f"\n✅ Done! Model at: https://huggingface.co/{OUTPUT_MODEL}")
    print(f"   Accuracy: {results['eval_accuracy']:.1%}")
    print(f"\n   Usage:")
    print(f"   from transformers import pipeline")
    print(f"   classifier = pipeline('audio-classification', model='{OUTPUT_MODEL}')")
    print(f"   result = classifier('audio.wav')")