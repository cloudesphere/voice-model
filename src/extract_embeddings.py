#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier


def find_wavs(input_dir: str):
    wavs = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    return wavs


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_audio_16k_mono(path: str):
    wav, sr = torchaudio.load(path)  # shape: [channels, time]
    if wav.size(0) > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav  # [1, time]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder of VAD wavs (16k mono preferred)")
    ap.add_argument("--output_dir", required=True, help="Where to write .npy embeddings")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--batch_size", type=int, default=1, help="Keep 1 on VPS CPU for stability")
    ap.add_argument("--min_sec", type=float, default=0.8, help="Skip segments shorter than this")
    args = ap.parse_args()

    ensure_dir(args.output_dir)

    device = torch.device(args.device)
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    model = model.to(device)
    model.eval()

    wavs = find_wavs(args.input_dir)
    if not wavs:
        raise SystemExit(f"No wav files found in: {args.input_dir}")

    kept = 0
    skipped = 0

    for i, f in enumerate(wavs, 1):
        # Load audio
        sig = load_audio_16k_mono(f)
        dur_sec = sig.shape[1] / 16000.0
        if dur_sec < args.min_sec:
            skipped += 1
            continue

        sig = sig.to(device)

        # SpeechBrain expects [batch, time] for encode_batch
        emb = model.encode_batch(sig.squeeze(0).unsqueeze(0))  # [1, 1, D] غالباً
        emb = emb.squeeze().detach().cpu().numpy().astype(np.float32)  # [D]

        out_name = Path(f).stem + ".npy"
        out_path = os.path.join(args.output_dir, out_name)
        np.save(out_path, emb)

        kept += 1
        if i % 200 == 0:
            print(f"[{i}/{len(wavs)}] kept={kept} skipped={skipped}")

    print("DONE")
    print("input_dir:", args.input_dir)
    print("output_dir:", args.output_dir)
    print("kept:", kept)
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
