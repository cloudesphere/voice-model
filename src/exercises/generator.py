import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

@dataclass
class GenerateConfig:
    out_dir: Path = Path("artifacts/exercises/generated_mvp")
    reader: str = "husary"
    segment_id: str = "seg_full_0001"
    ref_full_wav: str = "data/segments/husary_mvp/wav/seg_full_0001.wav"

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_exercise(obj: Dict[str, Any], out_path: Path) -> None:
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def generate_from_payload(
    payload: Dict[str, Any],
    *,
    config: GenerateConfig,
    user_full_wav: str,
    clips_map: Dict[str, str],
    timestamps_map: Dict[str, Any],
) -> Dict[str, Path]:
    """
    payload: result dict you already print (e.g., includes makharij_flags)
    clips_map: paths for ready-made clips (ref/user)
    timestamps_map: e.g., {"sirat_user":[26.0,26.94], "sirat_ref":[53.2,54.6], ...}
    Returns: {exercise_id: path}
    """
    _ensure_dir(config.out_dir)
    written = {}

    flags = payload.get("makharij_flags", []) or []
    # Example flag: {"type":"seen_as_sad_to_seen","expected":"صراط","got":"سراط"}
    for idx, f in enumerate(flags, start=1):
        if f.get("type") == "seen_as_sad_to_seen":
            ex = {
                "exercise_id": f"{config.reader}_auto_ex_sad_seen_{config.segment_id}_{idx:04d}",
                "type": "makharij",
                "difficulty": 1,
                "reader": config.reader,
                "segment_id": config.segment_id,
                "target": {
                    "confusion_set": ["ص","س"],
                    "expected_word": f.get("expected","صراط"),
                    "observed_word": f.get("got","سراط"),
                    "user_time_sec": timestamps_map.get("sirat_user"),
                    "ref_time_sec": timestamps_map.get("sirat_ref"),
                },
                "clips": {
                    "ref_clip": clips_map["ref_sirat"],
                    "user_clip": clips_map["user_sirat"],
                    "ref_control_clip": clips_map.get("ref_al_sirat"),
                    "user_control_clip": clips_map.get("user_al_sirat"),
                },
                "scoring": {
                    "method": "dtw_logmel_v1",
                    "note": "Generated automatically from makharij flag"
                },
                "feedback_ar": {
                    "summary": "تمرين تلقائي: لاحظنا خلطًا بين (ص) و(س) في (صراط).",
                    "tip": "كرر (صِرَاطَ) ببطء ثم بسرعة طبيعية، وركز على تفخيم الصاد."
                },
                "assets": {
                    "reference_full_wav": config.ref_full_wav,
                    "user_full_wav": user_full_wav
                }
            }
            out_path = config.out_dir / f"{ex['exercise_id']}.json"
            write_exercise(ex, out_path)
            written[ex["exercise_id"]] = out_path

    # Optional: if payload mentions merged "ولا/الضالين" as a phrase issue
    if any(f.get("type") == "merged_waw_la" for f in flags):
        ex = {
            "exercise_id": f"{config.reader}_auto_ex_phrase_wala_daalin_{config.segment_id}_0001",
            "type": "phrase",
            "difficulty": 1,
            "reader": config.reader,
            "segment_id": config.segment_id,
            "target": {
                "expected_phrase": "وَلَا الضَّالِّينَ",
                "user_time_sec": timestamps_map.get("waddaalin_user"),
                "ref_time_sec": timestamps_map.get("waddaalin_ref"),
            },
            "clips": {
                "ref_clip": clips_map["ref_wala_daalin"],
                "user_clip": clips_map["user_waddaalin"],
            },
            "scoring": {"method": "dtw_logmel_v1"},
            "feedback_ar": {
                "summary": "تمرين تلقائي: حاول فصل (ولا) عن (الضالين) وإظهار الشدة والمد.",
                "tip": "اقرأ (وَلَا) وحدها، ثم (الضَّالِّينَ) وحدها، ثم اجمعهما."
            },
            "assets": {
                "reference_full_wav": config.ref_full_wav,
                "user_full_wav": user_full_wav
            }
        }
        out_path = config.out_dir / f"{ex['exercise_id']}.json"
        write_exercise(ex, out_path)
        written[ex["exercise_id"]] = out_path

    return written
