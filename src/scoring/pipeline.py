from __future__ import annotations
import os
import json
import subprocess
from typing import Dict, Any, List, Optional

SCORES_DIR = os.path.join("artifacts", "scores")

def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout}\n\n"
            f"STDERR:\n{p.stderr}"
        )

def ensure_scores(user_wav: str, segment_id: str) -> Dict[str, Any]:
    os.makedirs(SCORES_DIR, exist_ok=True)

    word_out = os.path.join(SCORES_DIR, f"{segment_id}.word_score.json")
    makh_out = os.path.join(SCORES_DIR, f"{segment_id}.makharij_flags.json")
    dtw_out  = os.path.join(SCORES_DIR, f"{segment_id}.dtw_evidence.json")

    meta = {"generated": [], "cached": []}

    def build(env_key: str, out_path: str) -> Optional[List[str]]:
        tmpl = os.environ.get(env_key)
        if not tmpl:
            return None
        cmd = (tmpl
               .replace("{user_wav}", user_wav)
               .replace("{segment_id}", segment_id)
               .replace("{out_json}", out_path))
        return cmd.split()

    # WORD SCORE
    if not os.path.exists(word_out):
        cmd = build("WORD_SCORE_CMD", word_out)
        if not cmd:
            raise FileNotFoundError("WORD_SCORE_CMD not set")
        _run(cmd)
        meta["generated"].append("word_score")
    else:
        meta["cached"].append("word_score")

    # MAKHARIJ
    if not os.path.exists(makh_out):
        cmd = build("MAKHARIJ_CMD", makh_out)
        if not cmd:
            raise FileNotFoundError("MAKHARIJ_CMD not set")
        _run(cmd)
        meta["generated"].append("makharij_flags")
    else:
        meta["cached"].append("makharij_flags")

    # DTW (optional)
    if os.path.exists(dtw_out):
        meta["cached"].append("dtw_evidence")

    # Validate JSON
    json.load(open(word_out, "r", encoding="utf-8"))
    json.load(open(makh_out, "r", encoding="utf-8"))

    return meta
