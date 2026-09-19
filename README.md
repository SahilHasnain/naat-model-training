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

## Data

- `training-data/naat/*.wav` — 5-sec, 16kHz mono recitation chunks
- `training-data/explanation/*.wav` — 5-sec, 16kHz mono explanation chunks
- `training-data/manifest.json` — chunk metadata (`file`, `label`, `source`, `start`, `end`)