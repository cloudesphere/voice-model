from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from src.scoring.engine import score_audio
from src.scoring.pipeline import ensure_scores

EXERCISE_INDEX_PATH = os.path.join("artifacts", "exercise_index.json")
USER_SEGMENTS_DIR = os.path.join("artifacts", "user_segments")

app = FastAPI(title="Tajweed Scoring API (MVP)", version="0.1.0")


def ok(data: Any, meta: Optional[Dict[str, Any]] = None) -> JSONResponse:
    return JSONResponse(
        {
            "data": data,
            "error": None,
            "meta": meta or {},
        }
    )


def err(code: str, message: str, details: Optional[Dict[str, Any]] = None, status: int = 400) -> JSONResponse:
    return JSONResponse(
        {
            "data": None,
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            },
            "meta": {},
        },
        status_code=status,
    )


def load_index() -> Dict[str, Any]:
    if not os.path.exists(EXERCISE_INDEX_PATH):
        raise FileNotFoundError(f"exercise_index.json not found: {EXERCISE_INDEX_PATH}")
    with open(EXERCISE_INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/health")
def health():
    return ok({"status": "ok"})


@app.get("/exercise/random")
def exercise_random():
    try:
        idx = load_index()
        items = idx.get("exercises") or idx.get("items") or []
        if not isinstance(items, list) or not items:
            return err("NO_EXERCISES", "No exercises available in index", status=404)
        # simple random without extra deps
        import random
        ex = random.choice([x for x in items if isinstance(x, dict)] or items)
        return ok(ex, meta={"source": "index"})
    except Exception as e:
        return err("INDEX_ERROR", str(e), status=500)


@app.get("/exercise/{exercise_id}")
def exercise_get(exercise_id: str):
    try:
        idx = load_index()
        items = idx.get("exercises") or idx.get("items") or []
        if not isinstance(items, list) or not items:
            return err("NO_EXERCISES", "No exercises available in index", status=404)

        for ex in items:
            if isinstance(ex, dict) and (ex.get("id") == exercise_id or ex.get("exercise_id") == exercise_id):
                return ok(ex, meta={"source": "index"})

        return err("NOT_FOUND", f"Exercise not found: {exercise_id}", status=404)
    except Exception as e:
        return err("INDEX_ERROR", str(e), status=500)


@app.post("/exercise/score")
async def exercise_score(
    segment_id: str = Form(...),
    audio: UploadFile = File(...),
):
    t0 = time.time()
    try:
        # Save upload to canonical path
        os.makedirs(USER_SEGMENTS_DIR, exist_ok=True)
        out_path = os.path.join(USER_SEGMENTS_DIR, f"{segment_id}.user.wav")

        # NOTE: we accept wav for MVP. If m4a comes later we can convert via ffmpeg.
        content = await audio.read()
        if not content:
            return err("EMPTY_AUDIO", "Uploaded audio is empty", status=400)

        with open(out_path, "wb") as f:
            f.write(content)

        score_sources = ensure_scores(out_path, segment_id)
        result = score_audio(out_path, segment_id)
        dt_ms = int((time.time() - t0) * 1000)

        return ok(result, meta={"segment_id": segment_id, "saved_path": out_path, "time_ms": dt_ms, "score_sources": score_sources})
    except FileNotFoundError as e:
        return err("NOT_FOUND", str(e), status=404)
    except ValueError as e:
        return err("VALIDATION_ERROR", str(e), status=400)
    except Exception as e:
        msg = str(e)
        if "AUDIO_MISMATCH_SEGMENT" in msg:
            return err("VALIDATION_ERROR", msg, status=400)
        return err("INTERNAL_ERROR", msg, status=500)
