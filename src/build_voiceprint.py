#!/usr/bin/env python3
import argparse
import glob
import os
from pathlib import Path

import numpy as np


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True, help="Folder containing .npy embeddings")
    ap.add_argument("--out_path", required=True, help="Output voiceprint .npy path")
    ap.add_argument("--normalize_each", action="store_true", help="L2 normalize each embedding before averaging")
    args = ap.parse_args()

    embs = sorted(glob.glob(os.path.join(args.emb_dir, "*.npy")))
    if not embs:
        raise SystemExit(f"No .npy embeddings found in: {args.emb_dir}")

    vecs = []
    for f in embs:
        e = np.load(f).astype(np.float32)
        if args.normalize_each:
            e = l2_normalize(e)
        vecs.append(e)

    M = np.stack(vecs, axis=0)  # [N, D]
    voiceprint = M.mean(axis=0).astype(np.float32)
    voiceprint = l2_normalize(voiceprint)  # normalize final voiceprint (important for cosine)

    ensure_dir(str(Path(args.out_path).parent))
    np.save(args.out_path, voiceprint)

    print("DONE")
    print("emb_dir:", args.emb_dir)
    print("count:", M.shape[0])
    print("dim:", M.shape[1])
    print("out_path:", args.out_path)


if __name__ == "__main__":
    main()
