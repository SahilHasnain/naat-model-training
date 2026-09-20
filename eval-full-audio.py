"""
Run the app's exact inference logic against a trained model on a full audio file.

Mirrors apps/ai-service/classifier_fixed.py (per-second sliding-window voting,
explanation_threshold, merge_gap, confidence-crossing boundary refinement) so
results are comparable to what the app would produce, but loads a local model.

Usage:
  python eval-full-audio.py <audio.wav> [--model path-or-id] [--threshold 0.6]
"""

import argparse
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


class AudioClassifier:
    def __init__(self, model_name, threshold=0.6):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        self.chunk_duration = 5
        self.hop_duration = 1
        self.sample_rate = 16000
        self.merge_gap = 10
        self.explanation_threshold = threshold
        # The window's decision flips *before* the window is half-filled with the new
        # class (model is biased toward the dominant class). Empirically L/2=2.5s
        # overshoots; 1.5s lands nearest on known ground truth.
        self.boundary_offset = 1.5

        print(f"Loading model: {model_name}")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        id2label = self.model.config.id2label
        self.naat_idx = None
        self.expl_idx = None
        for idx_key, lbl in id2label.items():
            idx_int = int(idx_key) if isinstance(idx_key, str) else idx_key
            if lbl == "naat":
                self.naat_idx = idx_int
            else:
                self.expl_idx = idx_int
        if self.naat_idx is None:
            self.naat_idx = 0
        if self.expl_idx is None:
            self.expl_idx = 1
        print(f"Labels: {id2label}, naat_idx={self.naat_idx}, expl_idx={self.expl_idx}")

    def boundary_trajectory(self, audio, center, window, hop):
        """Fine-grained (2-class) probability trajectory around a boundary.
        Returns [(t, expl_prob, naat_prob), ...] using single classification windows."""
        sr = self.sample_rate
        n = len(audio)
        dur = n / sr
        last_start = dur - self.chunk_duration
        traj = []
        t = max(0.0, center - window)
        t_end = min(last_start, center + window)
        while t <= t_end:
            start = int(t * sr)
            a = audio[start : start + int(self.chunk_duration * sr)]
            if len(a) < sr:
                break
            if len(a) < self.chunk_duration * sr:
                a = np.pad(a, (0, self.chunk_duration * sr - len(a)))
            inputs = self.feature_extractor(
                a,
                sampling_rate=sr,
                max_length=self.chunk_duration * sr,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                probs = torch.softmax(self.model(**inputs).logits, dim=-1).squeeze().cpu().numpy()
            traj.append((t, float(probs[self.expl_idx]), float(probs[self.naat_idx])))
            t += hop
        return traj

    def _first_sustained_crossing(self, traj, wins_a_over_b, hyst=2):
        """Locate the exact probability 0.5-crossing (interpolated) where class A becomes
        dominant over B and stays dominant for >= hyst consecutive points.
        Returns the interpolated crossing time (pre half-window alignment)."""
        # dominance ratio r = pA / (pA + pB); crossing when r == 0.5
        for i in range(1, len(traj)):
            t0, e0, n0 = traj[i - 1]
            t1, e1, n1 = traj[i]
            r0 = e0 / (e0 + n0) if (e0 + n0) > 0 else 0.5
            r1 = e1 / (e1 + n1) if (e1 + n1) > 0 else 0.5
            if abs(r1 - r0) < 1e-9:
                continue
            # crossing of r == 0.5 between i-1 and i in the right direction?
            crossed_up = (not wins_a_over_b and r1 >= 0.5 > r0)
            crossed_down = (wins_a_over_b and r0 >= 0.5 > r1)
            if wins_a_over_b and not (r0 < 0.5 <= r1):
                continue
            if not wins_a_over_b and not (r0 >= 0.5 > r1):
                continue
            # confirm it is sustained for hyst points after i
            ok = True
            for j in range(i, min(i + hyst, len(traj))):
                tj, ej, nj = traj[j]
                rj = ej / (ej + nj) if (ej + nj) > 0 else 0.5
                if wins_a_over_b and rj < 0.5:
                    ok = False
                    break
                if not wins_a_over_b and rj > 0.5:
                    ok = False
                    break
            if ok:
                # linear interpolation of r(x) == 0.5
                frac = (0.5 - r0) / (r1 - r0)
                return t0 + frac * (t1 - t0)
        return None

    def find_conf_boundary(self, audio, rough_time, direction, window=3.0, hop=0.25, hyst=2):
        """Content-based boundary. The 5s window frames content at its leading edge, so a
        0.5-crossing in window-start time sits L/2 before the true content boundary:
        true_boundary = crossing + chunk_duration / 2."""
        traj = self.boundary_trajectory(audio, rough_time, window, hop)
        crossing = self._first_sustained_crossing(
            traj, wins_a_over_b=(direction == "start"), hyst=hyst
        )
        if crossing is None:
            # The coarse boundary may sit inside the explanation, so the window scan
            # never sees the pre-crossing side. Retry wider with no hysteresis.
            traj = self.boundary_trajectory(audio, rough_time, window + 2.0, hop)
            crossing = self._first_sustained_crossing(
                traj, wins_a_over_b=(direction == "start"), hyst=1
            )
        if crossing is None:
            print(f"  no confident crossing near {rough_time:.1f}s ({direction}) — keeping coarse boundary")
            return None
        refined = crossing + self.boundary_offset
        # Edge guard: only accept if comfortably inside the scanned window.
        lo, hi = rough_time - window + 0.5, rough_time + window + 0.5
        if not (lo <= refined <= hi):
            # Fallback: strongest probability gradient marks the content transition
            # even when there is no clean 0.5-crossing.
            grad = self._steepest_gradient(audio, rough_time)
            if grad is not None:
                print(f"  no clean crossing, using prob gradient {grad:.2f}s ({direction})")
                return grad
            print(f"  refined {refined:.2f}s outside window ({direction}) — keeping coarse boundary")
            return None
        return round(refined, 2)

    def _steepest_gradient(self, audio, center, window=3.0, hop=0.25):
        """Where explanation probability changes fastest — content-based transition point."""
        traj = self.boundary_trajectory(audio, center, window, hop)
        if len(traj) < 3:
            return None
        best = None
        best_g = -1.0
        for i in range(1, len(traj)):
            dt = traj[i][0] - traj[i - 1][0]
            if dt <= 0:
                continue
            g = abs(traj[i][1] - traj[i - 1][1]) / dt
            if g > best_g:
                best_g = g
                best = traj[i][0]
        return round(best + self.boundary_offset / 2.0, 2) if best is not None else None

    def classify_audio(self, audio):
        total_duration = len(audio) / self.sample_rate
        num_seconds = int(np.ceil(total_duration))

        naat_scores = np.zeros(num_seconds)
        expl_scores = np.zeros(num_seconds)
        vote_counts = np.zeros(num_seconds)

        num_windows = max(1, int(np.ceil((total_duration - self.chunk_duration) / self.hop_duration)) + 1)
        print(f"Classifying {num_windows} overlapping windows...")

        import time
        t0 = time.time()
        for i in range(num_windows):
            start = i * self.hop_duration
            end = start + self.chunk_duration
            start_sample = int(start * self.sample_rate)
            end_sample = min(int(end * self.sample_rate), len(audio))
            chunk_audio = audio[start_sample:end_sample]

            if len(chunk_audio) < self.sample_rate:
                break

            if len(chunk_audio) < self.chunk_duration * self.sample_rate:
                chunk_audio = np.pad(chunk_audio, (0, self.chunk_duration * self.sample_rate - len(chunk_audio)))

            inputs = self.feature_extractor(
                chunk_audio,
                sampling_rate=self.sample_rate,
                max_length=self.chunk_duration * self.sample_rate,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

            slot_start = int(start)
            slot_end = min(int(np.ceil(end)), num_seconds)
            for s in range(slot_start, slot_end):
                naat_scores[s] += probs[self.naat_idx]
                expl_scores[s] += probs[self.expl_idx]
                vote_counts[s] += 1

            if (i + 1) % 100 == 0:
                el = time.time() - t0
                print(f"  {i+1}/{num_windows} ({el:.0f}s elapsed)")

        chunks = []
        for s in range(num_seconds):
            if vote_counts[s] == 0:
                continue
            avg_naat = naat_scores[s] / vote_counts[s]
            avg_expl = expl_scores[s] / vote_counts[s]
            if avg_expl >= avg_naat and avg_expl >= self.explanation_threshold:
                label = "explanation"
            else:
                label = "naat"
            score = max(avg_naat, avg_expl)
            sec_end = min(s + 1, total_duration)
            chunks.append({
                "start": round(float(s), 2),
                "end": round(sec_end, 2),
                "label": label,
                "score": round(float(score), 4),
            })

        runs = []
        if chunks:
            cur = {"start": chunks[0]["start"], "end": chunks[0]["end"],
                   "label": chunks[0]["label"], "scores": [chunks[0]["score"]]}
            for c in chunks[1:]:
                if c["label"] == cur["label"]:
                    cur["end"] = c["end"]
                    cur["scores"].append(c["score"])
                else:
                    runs.append(cur)
                    cur = {"start": c["start"], "end": c["end"],
                           "label": c["label"], "scores": [c["score"]]}
            runs.append(cur)

        merged = []
        for r in runs:
            if r["label"] == "explanation":
                if (merged and merged[-1]["label"] == "explanation"
                        and r["start"] - merged[-1]["end"] <= self.merge_gap):
                    merged[-1]["end"] = r["end"]
                    merged[-1]["scores"].extend(r["scores"])
                else:
                    merged.append(dict(r))
            else:
                merged.append(dict(r))

        print("Refining segment boundaries (confidence crossing)...")
        for seg in merged:
            if seg["label"] == "explanation":
                if seg["start"] > 0:
                    refined = self.find_conf_boundary(audio, seg["start"], "start")
                    if refined is not None:
                        seg["start"] = refined
                if seg["end"] < total_duration:
                    refined = self.find_conf_boundary(audio, seg["end"], "end")
                    if refined is not None:
                        seg["end"] = refined

        speech_segments = []
        for seg in merged:
            if seg["label"] == "explanation":
                dur = seg["end"] - seg["start"]
                if dur > 5:
                    speech_segments.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "confidence": round(sum(seg["scores"]) / len(seg["scores"]), 4),
                        "duration": round(dur),
                    })

        total_speech = sum(s["end"] - s["start"] for s in merged if s["label"] == "explanation")
        total_singing = total_duration - total_speech

        return {
            "duration": round(total_duration, 2),
            "speechSegments": speech_segments,
            "totalSpeechDuration": round(total_speech),
            "totalSingingDuration": round(total_singing),
        }


def fmt(t):
    m = int(t) // 60
    s = int(t) % 60
    return f"{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="naat-classifier-checkpoints/final-model")
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--max-sec", type=float, default=None, help="Only process the first N seconds")
    args = ap.parse_args()

    print(f"Reading {args.audio}...")
    audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if sr != 16000:
        print(f"Resampling {sr} -> 16000")
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = audio[: int(args.max_sec * 16000)] if args.max_sec else audio

    clf = AudioClassifier(args.model, threshold=args.threshold)
    result = clf.classify_audio(audio)

    print("\n==== RESULT ====")
    print(f"Duration: {result['duration']}s "
          f"({fmt(result['duration'])})")
    print(f"Speech: {result['totalSpeechDuration']}s "
          f"({fmt(result['totalSpeechDuration'])}) | "
          f"Singing: {result['totalSingingDuration']}s")
    print("\nSpeech segments:")
    for s in result["speechSegments"]:
        print(f"  {fmt(s['start'])} - {fmt(s['end'])}  "
              f"({s['duration']}s)  conf={s['confidence']}")


if __name__ == "__main__":
    main()