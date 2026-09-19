# Naat Model Training

Trains the **Naat vs Explanation** audio classifier (Wav2Vec2 fine-tune) from static,
pre-exported training data. No Appwrite dependency — the labeled 5-second WAV chunks live
in this repo under `training-data/`.

## Pipeline

1. **Static data** — `training-data/` contains labeled WAV chunks + `manifest.json`
   (`naat/` = recitation, `explanation/` = speech/commentary). Already exported.
2. **Upload to HuggingFace** — `upload-to-huggingface.py` pushes the static data to the
   `naat-classifier` dataset repo on HuggingFace Hub.
3. **Fine-tune** — `colab-finetune.py` loads the dataset from HuggingFace, fine-tunes
   `facebook/wav2vec2-base`, and pushes the model to `naat-classifier-model`.

The resulting model is consumed by the app at `sahilhasnain07/naat-classifier-model`.

## Retrain steps

**Mode A — GPU / Colab (full fine-tune, default):**

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Login & upload static data to HF dataset
huggingface-cli login
python upload-to-huggingface.py

# 3. Fine-tune (T4 GPU recommended) — Colab or GPU machine
python colab-finetune.py
# pushes model to huggingface.co/<user>/naat-classifier-model

# 4. Update the app's local model cache so it downloads the new model
rm -rf ~/.cache/huggingface/hub/models--sahilhasnain07--naat-classifier-model
```

**Mode B — Local CPU (no HuggingFace dataset needed):**

Reads `training-data/` straight from disk — the HF dataset repo, upload step, and
even the login are all skipped unless you push at the end.

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Train on CPU, frozen backbone (fast) — saves locally, no push
python colab-finetune.py --local --freeze-base --no-push

# 2b. Alternatively train the full model on CPU (slow, hours):
python colab-finetune.py --local --no-push
```

Useful flags: `--data-dir` (default `training-data`), `--epochs`, `--batch-size`,
`--output-dir`. Drop `--no-push` to push the result to
`huggingface.co/<user>/naat-classifier-model` (requires `huggingface-cli login`).

`--freeze-base` freezes the wav2vec2 backbone and trains only the head — that cuts
trainable params from ~95M to ~a few hundred thousand, making CPU runs take minutes
instead of hours. With only 44 chunks, the frozen-head result is typically at least
as accurate as full fine-tuning (less overfitting).

## Data

- `training-data/naat/*.wav` — 5-sec, 16kHz mono recitation chunks
- `training-data/explanation/*.wav` — 5-sec, 16kHz mono explanation chunks
- `training-data/manifest.json` — chunk metadata (`file`, `label`, `source`, `start`, `end`)