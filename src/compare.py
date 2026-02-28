#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
compare.py - MVP voice similarity comparison:
- Loads prebuilt voiceprints (husary/minshawi) from artifacts/voiceprints/*.npy
- Extracts ECAPA embedding from user audio
- Computes cosine similarity and returns closest reader + scores
- Applies threshold + margin decision policy

Stack:
- Python 3.12
- torch/torchaudio 2.8.0
- speechbrain 1.0.3
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torchaudio

from speechbrain.pretrained import EncoderClassifier


# ---------------------------
# Utilities
# ---------------------------

def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.reshape(-1).astype(np.float32)
    n = float(np.linalg.norm(x))
    return x / (n + eps)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def load_voiceprints(voiceprints_dir: str) -> Dict[str, np.ndarray]:
    if not os.path.isdir(voiceprints_dir):
        raise FileNotFoundError(f"Voiceprints dir not found: {voiceprints_dir}")

    voiceprints = {}
    for fname in os.listdir(voiceprints_dir):
        if fname.endswith(".npy"):
            name = os.path.splitext(fname)[0]
            path = os.path.join(voiceprints_dir, fname)
            vp = np.load(path)
            vp = _l2_normalize(vp)
            voiceprints[name] = vp

    if not voiceprints:
        raise RuntimeError("No voiceprints found.")

    dims = {k: v.shape[0] for k, v in voiceprints.items()}
    if len(set(dims.values())) != 1:
        raise ValueError(f"Voiceprints dimensions mismatch: {dims}")

    return voiceprints


def load_audio_mono_16k(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    wav, sr = torchaudio.load(audio_path)
    wav = wav.float()

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav


# ---------------------------
# Result dataclass
# ---------------------------

@dataclass
class CompareResult:
    closest: str
    closest_score: float
    second_best: str
    second_score: float
    margin: float
    pass_threshold: bool
    pass_margin: bool
    decision: str
    scores: Dict[str, float]
    threshold: float
    margin_threshold: float


# ---------------------------
# ECAPA embedding extraction
# ---------------------------

class ECAPAEmbedder:
    def __init__(
        self,
        device: str = "cpu",
        model_source: str = "speechbrain/spkrec-ecapa-voxceleb",
        cache_dir: Optional[str] = None,
    ):
        self.device = torch.device(device)
        run_opts = {"device": str(self.device)}
        savedir = cache_dir or "artifacts/speechbrain_models/ecapa"
        os.makedirs(savedir, exist_ok=True)

        self.model = EncoderClassifier.from_hparams(
            source=model_source,
            savedir=savedir,
            run_opts=run_opts,
        )

    @torch.no_grad()
    def embed(self, wav_16k_mono: torch.Tensor) -> np.ndarray:
        wav_16k_mono = wav_16k_mono.to(self.device)
        emb = self.model.encode_batch(wav_16k_mono)
        emb = emb.squeeze().detach().cpu().numpy()
        return _l2_normalize(emb)


# ---------------------------
# Core comparison logic
# ---------------------------

def compare_embedding_to_voiceprints(
    emb: np.ndarray,
    voiceprints: Dict[str, np.ndarray],
    threshold: float,
    margin_threshold: float,
) -> CompareResult:

    scores = {k: cosine_similarity(emb, vp) for k, vp in voiceprints.items()}

    sorted_scores = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = sorted_scores[0]
    second_name, second_score = (
        sorted_scores[1] if len(sorted_scores) > 1 else ("", float("-inf"))
    )

    margin_value = best_score - second_score
    pass_threshold = best_score >= threshold
    pass_margin = margin_value >= margin_threshold

    decision = "MATCH" if (pass_threshold and pass_margin) else "NO_MATCH"

    return CompareResult(
        closest=best_name,
        closest_score=float(best_score),
        second_best=second_name,
        second_score=float(second_score),
        margin=float(margin_value),
        pass_threshold=pass_threshold,
        pass_margin=pass_margin,
        decision=decision,
        scores={k: float(v) for k, v in sorted(scores.items())},
        threshold=float(threshold),
        margin_threshold=float(margin_threshold),
    )


# ---------------------------
# CLI
# ---------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare user voice against predefined voiceprints using ECAPA + cosine."
    )
    p.add_argument("audio", help="Path to user audio file.")
    p.add_argument("--voiceprints-dir", default="artifacts/voiceprints")
    p.add_argument("--threshold", type=float, default=0.54)
    p.add_argument("--margin", type=float, default=0.20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-source", default="speechbrain/spkrec-ecapa-voxceleb")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--json", action="store_true")
    return p


def main():
    args = build_arg_parser().parse_args()

    voiceprints = load_voiceprints(args.voiceprints_dir)

    embedder = ECAPAEmbedder(
        device=args.device,
        model_source=args.model_source,
        cache_dir=args.cache_dir,
    )

    wav = load_audio_mono_16k(args.audio)
    emb = embedder.embed(wav)

    res = compare_embedding_to_voiceprints(
        emb=emb,
        voiceprints=voiceprints,
        threshold=args.threshold,
        margin_threshold=args.margin,
    )

    payload = {
        "input_audio": os.path.abspath(args.audio),
        "closest_reader": res.closest,
        "closest_score": round(res.closest_score, 6),
        "second_best": {
            "reader": res.second_best,
            "score": round(res.second_score, 6),
        },
        "margin": round(res.margin, 6),
        "threshold": res.threshold,
        "margin_threshold": res.margin_threshold,
        "pass_threshold": res.pass_threshold,
        "pass_margin": res.pass_margin,
        "decision": res.decision,
        "scores": {k: round(v, 6) for k, v in res.scores.items()},
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print("✅ Comparison Result")
        print(f"- input: {payload['input_audio']}")
        print(f"- closest: {payload['closest_reader']} (score={payload['closest_score']})")
        print(f"- second_best: {payload['second_best']['reader']} (score={payload['second_best']['score']})")
        print(f"- margin: {payload['margin']}")
        print(f"- pass_threshold: {payload['pass_threshold']} (threshold={payload['threshold']})")
        print(f"- pass_margin: {payload['pass_margin']} (margin_threshold={payload['margin_threshold']})")
        print(f"- decision: {payload['decision']}")
        print("- scores:")
        for k, v in payload["scores"].items():
            print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
