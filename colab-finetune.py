"""
Naat vs Explanation Classifier — Fine-tune Wav2Vec2 on Google Colab

Instructions:
  1. Open Google Colab (https://colab.research.google.com)
  2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
  3. Copy-paste this entire file into a single cell and run it
  4. When prompted, paste your HuggingFace token (needs write access)
  5. Training takes ~5-10 min on T4 with 97 samples
  6. Model is pushed to: huggingface.co/<your-user>/naat-classifier-model
"""

# ── 1. Install dependencies ──────────────────────────────────
print("📦 Installing dependencies...")
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "datasets", "transformers", "accelerate", "evaluate",
    "huggingface_hub", "soundfile", "librosa"])

# ── 2. Login to HuggingFace ──────────────────────────────────
from huggingface_hub import login, HfApi
print("\n🔑 Logging in to HuggingFace...")
login()  # Will prompt for token interactively — never hardcode tokens!

api = HfApi()
username = api.whoami()["name"]
print(f"   Logged in as: {username}")

# ── 3. Load dataset ──────────────────────────────────────────
from datasets import load_dataset, Audio, ClassLabel
print("\n📋 Loading dataset...")

dataset = load_dataset(
    f"{username}/naat-classifier",
    revision="main",
    download_mode="force_redownload",
)

# Ensure audio column is 16kHz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Cast label from string to ClassLabel
dataset = dataset["train"].cast_column("label", ClassLabel(names=["naat", "explanation"]))

print(f"   Total samples: {len(dataset)}")
print(f"   Features: {dataset.features}")

# ── 4. Train/test split ──────────────────────────────────────
dataset = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
print(f"   Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")

# ── 5. Load feature extractor & model ────────────────────────
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_NAME = "facebook/wav2vec2-base"
OUTPUT_MODEL = f"{username}/naat-classifier-model"

print(f"\n🧠 Loading {MODEL_NAME}...")
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)

label2id = {"naat": "0", "explanation": "1"}
id2label = {"0": "naat", "1": "explanation"}

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    label2id=label2id,
    id2label=id2label,
)

# ── 6. Preprocess (batched, following official HF guide) ─────
def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=16000,
        max_length=16000 * 5,  # 5 seconds
        truncation=True,
    )
    return inputs

print("⚙️  Preprocessing audio...")
encoded = dataset.map(preprocess_function, remove_columns="audio", batched=True)

# ── 7. Training setup ────────────────────────────────────────
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer

accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)

training_args = TrainingArguments(
    output_dir="./naat-classifier-checkpoints",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=True,
    hub_model_id=OUTPUT_MODEL,
    hub_private_repo=True,
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

# ── 10. Push to HuggingFace ──────────────────────────────────
print(f"\n📤 Pushing model to {OUTPUT_MODEL}...")
trainer.push_to_hub()
feature_extractor.push_to_hub(OUTPUT_MODEL)

print(f"\n✅ Done! Model at: https://huggingface.co/{OUTPUT_MODEL}")
print(f"   Accuracy: {results['eval_accuracy']:.1%}")
print(f"\n   Usage:")
print(f"   from transformers import pipeline")
print(f"   classifier = pipeline('audio-classification', model='{OUTPUT_MODEL}')")
print(f"   result = classifier('audio.wav')")
