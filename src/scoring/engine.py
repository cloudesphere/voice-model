# src/scoring/engine.py
# Unified Scoring Engine (MVP)
#
# - score_audio(user_wav_path, segment_id) -> dict JSON
# - Ensures required score artifacts exist by running scorer CLIs (WORD_SCORE_CMD, MAKHARIJ_CMD, optional DTW_CMD)
# - Loads alignment for QC only (does not re-run alignment here)
# - Filters DTW evidence to "important" only
# - Recommends exercises from hydrated exercise_index.json (or falls back gracefully)
#
# ENV (recommended):
#   export WORD_SCORE_CMD='python3 -m src.scoring.word_scorer_cli --user_wav {user_wav} --segment_id {segment_id} --out {out}'
#   export MAKHARIJ_CMD='python3 -m src.scoring.makharij_cli --word_score {word_score} --out {out}'
#   export DTW_CMD='python3 -m src.scoring.dtw_cli --segment_id {segment_id} --out {out}'   # optional
#   export TAJWEED_ARTIFACTS_DIR=/path/to/artifacts   # optional; default "artifacts" (relative to cwd)
#
# Usage:
#   python3 -m src.scoring.engine --user_wav <wav> --segment_id seg_full_0001 --pretty

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from typing import Any, Dict, List, Optional

# Project paths (override with env TAJWEED_ARTIFACTS_DIR for VPS/deploy)
ARTIFACTS_DIR = os.path.abspath(
    os.path.expanduser(os.environ.get("TAJWEED_ARTIFACTS_DIR", "artifacts"))
)
ALIGNMENTS_DIR = os.path.join(ARTIFACTS_DIR, "alignments")
SCORES_DIR = os.path.join(ARTIFACTS_DIR, "scores")

# Exercises
EXERCISE_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "exercise_index.json")
DEFAULT_TOP_EXERCISES = 3

# DTW: always include evidence for "high" severity flags, plus these confusions
DTW_IMPORTANT_CONFUSIONS = {
    "ص↔س",
    # extend later: "ض↔ظ", "ت↔ط", ...
}


# ----------------------------
# Basic helpers
# ----------------------------
def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _load_json(path: str, label: str) -> Any:
    _assert(os.path.exists(path), f"{label} not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def make_json_safe(x: Any) -> Any:
    """
    Convert non-JSON-safe types to JSON-safe equivalents (best-effort).
    """
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, dict):
        return {str(k): make_json_safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [make_json_safe(v) for v in x]
    # fallback
    return str(x)


# ----------------------------
# Artifact paths
# ----------------------------
def alignment_path(segment_id: str) -> str:
    return os.path.join(ALIGNMENTS_DIR, f"{segment_id}.alignment.json")


def _artifacts_dir_candidates() -> list:
    """Candidate artifact root dirs: default, then sibling voice-model/artifacts (for VPS voice-model-code layout)."""
    cwd = os.getcwd()
    candidates = [ARTIFACTS_DIR]
    sibling = os.path.abspath(os.path.join(cwd, "..", "voice-model", "artifacts"))
    if sibling != ARTIFACTS_DIR and sibling not in candidates:
        candidates.append(sibling)
    return candidates


def _resolve_artifacts_dir(segment_id: str) -> str:
    """Return the first artifacts dir that contains the segment alignment file. Updates globals and env when falling back."""
    global ARTIFACTS_DIR, ALIGNMENTS_DIR, SCORES_DIR, EXERCISE_INDEX_PATH
    for base in _artifacts_dir_candidates():
        aln_p = os.path.join(base, "alignments", f"{segment_id}.alignment.json")
        if os.path.exists(aln_p):
            if base != ARTIFACTS_DIR:
                os.environ["TAJWEED_ARTIFACTS_DIR"] = base
                ARTIFACTS_DIR = base
                ALIGNMENTS_DIR = os.path.join(ARTIFACTS_DIR, "alignments")
                SCORES_DIR = os.path.join(ARTIFACTS_DIR, "scores")
                EXERCISE_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "exercise_index.json")
            return base
    return ARTIFACTS_DIR  # no change, will fail in _load_json with clear error


def score_path(segment_id: str, kind: str) -> str:
    # kind: word_score | makharij_flags | dtw_evidence
    return os.path.join(SCORES_DIR, f"{segment_id}.{kind}.json")


# ----------------------------
# QC for alignment (MVP)
# ----------------------------
def qc_alignment(aln: Dict[str, Any]) -> None:
    """
    Minimal QC:
      - must contain a list-like field representing words/tokens with timestamps
      - monotonic start/end
      - non-zero spans
    We accept different key names to keep it robust.
    """
    words = (
        aln.get("words")
        or aln.get("word_timestamps")
        or aln.get("tokens")
        or aln.get("items")
    )
    _assert(words is not None, "alignment.json missing words/word_timestamps/tokens field")
    _assert(isinstance(words, list), "alignment words must be a list")

    prev_end = -1e9
    for i, w in enumerate(words):
        if not isinstance(w, dict):
            continue
        s = w.get("start")
        e = w.get("end")
        if s is None or e is None:
            continue
        try:
            s = float(s)
            e = float(e)
        except Exception:
            raise ValueError(f"alignment word[{i}] start/end not numeric")

        _assert(e >= s, f"alignment word[{i}] end < start")
        _assert((e - s) > 0.0, f"alignment word[{i}] zero span")
        _assert(s >= prev_end - 1e-6, f"alignment non-monotonic at word[{i}]")
        prev_end = e


# ----------------------------
# QC for word_score
# ----------------------------
def qc_word_score(ws: Dict[str, Any]) -> None:
    _assert(isinstance(ws, dict), "word_score must be a JSON object")
    _assert("wer" in ws, "word_score missing 'wer'")
    wer = float(ws["wer"])
    _assert(0.0 <= wer <= 1.0, f"word_score.wer out of range: {wer}")
    for k in ("matches", "subs", "dels", "ins"):
        if k in ws:
            _assert(float(ws[k]) >= 0, f"word_score.{k} negative")
    if "ops" in ws:
        _assert(isinstance(ws["ops"], list), "word_score.ops must be list")


# ----------------------------
# Command runner + ensure_scores
# ----------------------------
def _format_cmd(
    tpl: str,
    *,
    user_wav: str,
    segment_id: str,
    out: str,
    word_score: str = "",
) -> List[str]:
    cmd = (tpl or "").format(
        user_wav=user_wav,
        segment_id=segment_id,
        out=out,
        word_score=word_score,
    ).strip()
    _assert(cmd, "command template is empty")
    return shlex.split(cmd)


def _run_cmd(
    tpl: str,
    *,
    user_wav: str,
    segment_id: str,
    out: str,
    word_score: str = "",
) -> None:
    cmd = _format_cmd(tpl, user_wav=user_wav, segment_id=segment_id, out=out, word_score=word_score)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + (r.stdout or "")
            + "\n\nSTDERR:\n"
            + (r.stderr or "")
        )


def ensure_scores(user_wav: str, segment_id: str) -> Dict[str, List[str]]:
    """
    Ensure required scorer outputs exist under artifacts/scores.

    Required:
      - word_score.json via WORD_SCORE_CMD
      - makharij_flags.json via MAKHARIJ_CMD (needs {word_score})

    Optional:
      - dtw_evidence.json via DTW_CMD
    """
    os.makedirs(SCORES_DIR, exist_ok=True)

    ws_p = score_path(segment_id, "word_score")
    mk_p = score_path(segment_id, "makharij_flags")
    dtw_p = score_path(segment_id, "dtw_evidence")

    generated: List[str] = []
    cached: List[str] = []

    if not os.path.exists(ws_p):
        tpl = os.environ.get("WORD_SCORE_CMD", "").strip()
        _assert(tpl, "WORD_SCORE_CMD not set")
        _run_cmd(tpl, user_wav=user_wav, segment_id=segment_id, out=ws_p)
        generated.append("word_score")
    else:
        cached.append("word_score")

    if not os.path.exists(mk_p):
        tpl = os.environ.get("MAKHARIJ_CMD", "").strip()
        _assert(tpl, "MAKHARIJ_CMD not set")
        _run_cmd(tpl, user_wav=user_wav, segment_id=segment_id, out=mk_p, word_score=ws_p)
        generated.append("makharij_flags")
    else:
        cached.append("makharij_flags")

    tpl = os.environ.get("DTW_CMD", "").strip()
    if tpl:
        if not os.path.exists(dtw_p):
            _run_cmd(tpl, user_wav=user_wav, segment_id=segment_id, out=dtw_p)
            generated.append("dtw_evidence")
        else:
            cached.append("dtw_evidence")

    return {"generated": generated, "cached": cached}


# ----------------------------
# Overall score (MVP)
# ----------------------------
def compute_overall_score(word_score: Dict[str, Any], makharij_flags: List[Dict[str, Any]]) -> float:
    """
    MVP scoring:
      base = (1 - WER) * 100
      penalties for makharij flags by severity
    """
    wer = float(word_score.get("wer", 1.0))
    base = max(0.0, min(100.0, (1.0 - wer) * 100.0))

    penalty = 0.0
    for f in makharij_flags or []:
        sev = (f.get("severity") or "low").lower()
        if sev == "high":
            penalty += 12.0
        elif sev == "medium":
            penalty += 6.0
        else:
            penalty += 3.0

    return float(max(0.0, min(100.0, base - penalty)))


# ----------------------------
# DTW filtering (important only)
# ----------------------------
def filter_important_dtw(
    makharij_flags: List[Dict[str, Any]],
    dtw_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    important_flag_ids: set[str] = set()
    for f in makharij_flags or []:
        fid = f.get("id")
        sev = (f.get("severity") or "low").lower()
        conf = (f.get("confusion") or "").strip()
        if fid and (sev == "high" or conf in DTW_IMPORTANT_CONFUSIONS):
            important_flag_ids.add(str(fid))

    out: List[Dict[str, Any]] = []
    for r in dtw_rows or []:
        if not isinstance(r, dict):
            continue
        fid = r.get("flag_id")
        if fid and str(fid) in important_flag_ids:
            rr = dict(r)
            rr["important"] = True
            out.append(rr)

    # Safety: clamp dtw_score to [0,100] if present
    for r in out:
        s = r.get("dtw_score")
        if s is not None:
            try:
                fs = float(s)
                r["dtw_score"] = max(0.0, min(100.0, fs))
            except Exception:
                pass

    return make_json_safe(out)


# ----------------------------
# Exercises recommendation
# ----------------------------
def _try_load_exercise_store():
    """
    Lazy import to avoid hard failures if exercises module changes.
    """
    try:
        from src.exercises.store import ExerciseStore  # type: ignore
        return ExerciseStore
    except Exception:
        return None


def recommend_exercises(makharij_flags: List[Dict[str, Any]], top_k: int = DEFAULT_TOP_EXERCISES) -> List[Dict[str, Any]]:
    """
    Strategy:
      1) If no flags => return []
      2) Load ExerciseStore and use search_by_flags (if available)
      3) Return "thin" exercise objects (id,type,title,prompt,targets)
    """
    flags = makharij_flags or []
    if not flags:
        return []

    ExerciseStore = _try_load_exercise_store()
    if ExerciseStore is None:
        return []

    try:
        store = ExerciseStore(EXERCISE_INDEX_PATH)  # type: ignore
    except Exception:
        return []

    selected: List[Dict[str, Any]] = []
    try:
        if hasattr(store, "search_by_flags"):
            selected = store.search_by_flags(flags, limit=top_k)  # type: ignore
        elif hasattr(store, "search"):
            # fallback: try a generic search API
            selected = store.search(flags, limit=top_k)  # type: ignore
    except Exception:
        selected = []

    # normalize output
    out: List[Dict[str, Any]] = []
    for ex in selected[:top_k]:
        if not isinstance(ex, dict):
            continue
        out.append(
            {
                "id": ex.get("id") or ex.get("exercise_id"),
                "type": ex.get("type"),
                "title": ex.get("title"),
                "prompt": ex.get("prompt"),
                "targets": ex.get("targets") or [],
            }
        )

    # Drop invalid ids
    out = [e for e in out if e.get("id")]
    return make_json_safe(out)


# ----------------------------
# Main API
# ----------------------------
def score_audio(user_wav_path: str, segment_id: str) -> Dict[str, Any]:
    user_wav_path = os.path.abspath(os.path.expanduser(user_wav_path))
    _assert(
        os.path.exists(user_wav_path),
        f"user wav not found: {user_wav_path} (check the file exists, e.g. ls -la {user_wav_path!r})",
    )
    _assert(segment_id and isinstance(segment_id, str), "segment_id required")

    # Resolve artifacts dir (default or fallback e.g. ../voice-model/artifacts when run from voice-model-code)
    _resolve_artifacts_dir(segment_id)

    # Alignment QC (reference must exist)
    aln_p = alignment_path(segment_id)
    aln = _load_json(aln_p, "alignment")
    _assert(isinstance(aln, dict), "alignment must be a JSON object")
    qc_alignment(aln)

    # Ensure scorer outputs exist
    score_sources = ensure_scores(user_wav_path, segment_id)

    # Load required outputs
    word_p = score_path(segment_id, "word_score")
    flags_p = score_path(segment_id, "makharij_flags")
    dtw_p = score_path(segment_id, "dtw_evidence")

    word_score = _load_json(word_p, "word_score")
    qc_word_score(word_score)

    makharij_flags = _load_json(flags_p, "makharij_flags")
    _assert(isinstance(makharij_flags, list), "makharij_flags must be a JSON array")

    # Optional DTW
    dtw_rows: List[Dict[str, Any]] = []
    if os.path.exists(dtw_p):
        dtw_obj = _load_json(dtw_p, "dtw_evidence")
        if isinstance(dtw_obj, list):
            dtw_rows = dtw_obj
        elif isinstance(dtw_obj, dict) and isinstance(dtw_obj.get("rows"), list):
            dtw_rows = dtw_obj["rows"]

    # Filter DTW to important only
    dtw_evidence = filter_important_dtw(makharij_flags, dtw_rows)

    # Recommend exercises (only if flags exist)
    recommended = recommend_exercises(makharij_flags, top_k=DEFAULT_TOP_EXERCISES)

    overall = compute_overall_score(word_score, makharij_flags)

    out = {
        "overall_score": overall,
        "word_score": make_json_safe(word_score),
        "makharij_flags": make_json_safe(makharij_flags),
        "dtw_evidence": dtw_evidence,
        "recommended_exercises": recommended,
        "meta": {
            "segment_id": segment_id,
            "score_sources": score_sources,  # {generated:[], cached:[]}
        },
    }

    # Final QC (shape)
    for k in ["overall_score", "word_score", "makharij_flags", "dtw_evidence", "recommended_exercises", "meta"]:
        _assert(k in out, f"output missing key: {k}")
    _assert(isinstance(out["makharij_flags"], list), "makharij_flags must be list")
    _assert(isinstance(out["dtw_evidence"], list), "dtw_evidence must be list")
    _assert(isinstance(out["recommended_exercises"], list), "recommended_exercises must be list")

    return out


# ----------------------------
# CLI
# ----------------------------
def _main() -> None:
    global ARTIFACTS_DIR, ALIGNMENTS_DIR, SCORES_DIR, EXERCISE_INDEX_PATH
    ap = argparse.ArgumentParser(description="Unified Tajweed Scoring Engine (MVP)")
    ap.add_argument("--user_wav", required=True, help="User WAV path (16kHz mono PCM recommended)")
    ap.add_argument("--segment_id", required=True, help="Segment id (e.g., seg_full_0001)")
    ap.add_argument(
        "--artifacts-dir",
        default=None,
        help="Path to artifacts dir (alignments, scores). Overrides TAJWEED_ARTIFACTS_DIR. Example: /root/voice-model/artifacts",
    )
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    ap.add_argument("--out", default=None, help="Optional output JSON path")
    args = ap.parse_args()

    if args.artifacts_dir:
        args.artifacts_dir = os.path.abspath(os.path.expanduser(args.artifacts_dir))
        os.environ["TAJWEED_ARTIFACTS_DIR"] = args.artifacts_dir
        ARTIFACTS_DIR = args.artifacts_dir
        ALIGNMENTS_DIR = os.path.join(ARTIFACTS_DIR, "alignments")
        SCORES_DIR = os.path.join(ARTIFACTS_DIR, "scores")
        EXERCISE_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "exercise_index.json")

    result = score_audio(args.user_wav, args.segment_id)
    if args.out:
        _dump_json(args.out, result)

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    _main()
