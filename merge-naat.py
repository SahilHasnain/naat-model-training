"""
Cut the detected explanation (speech) segments out of a full audio and merge the
remaining naat into one listenable file.

Usage:
  python merge-naat.py <audio> --segments segments.json -o naat-merged.wav

segments.json: {"speechSegments": [{"start": s, "end": e, ...}, ...]}
"""

import argparse
import json
import sys

import numpy as np
import soundfile as sf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--segments", required=True, help="JSON with speechSegments list")
    ap.add_argument("-o", "--output", default="naat-merged.wav")
    args = ap.parse_args()

    with open(args.segments, encoding="utf-8") as f:
        data = json.load(f)
    segs = data["speechSegments"]
    print(f"{len(segs)} speech segments to remove")

    audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    total = n / sr
    print(f"Loaded {args.audio}: {total:.1f}s @ {sr}Hz")

    # Build per-sample keep mask.
    keep = np.ones(n, dtype=bool)
    for s in segs:
        a = int(round(s["start"] * sr))
        b = int(round(s["end"] * sr))
        # trim the excess-carved explanations at boundaries by a small margin
        # is NOT applied; we trust the refined boundaries from the classifier.
        a, b = max(0, a), min(n, b)
        if b > a:
            keep[a:b] = False

    removed_s = int(np.count_nonzero(~keep)) / sr
    kept_s = int(np.count_nonzero(keep)) / sr
    print(f"Removing {removed_s:.1f}s of speech, keeping {kept_s:.1f}s of naat")

    if kept_s <= 1:
        print("Error: nothing left after removing speech — check segment timestamps")
        sys.exit(1)

    # Extract & concatenate the kept regions (this is the actual merge).
    seqs = []
    idx = 0
    while idx < n:
        if keep[idx]:
            j = idx
            while j < n and keep[j]:
                j += 1
            seqs.append(audio[idx:j])
            idx = j
        else:
            idx += 1

    merged = np.concatenate(seqs) if seqs else np.zeros(0, dtype=np.float32)
    sf.write(args.output, merged, sr, subtype="PCM_16")
    print(f"Wrote {args.output}: {len(merged)/sr:.1f}s merged naat @ {sr}Hz")


if __name__ == "__main__":
    main()