from __future__ import annotations
import argparse, json, os
from typing import List, Dict, Any

CONFUSIONS = {
    ("ص", "س"): ("ص↔س", "high"),
    ("س", "ص"): ("ص↔س", "high"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--word_score", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ws = json.load(open(args.word_score, "r", encoding="utf-8"))
    ops: List[Dict[str, Any]] = ws.get("ops", []) or []

    flags = []
    fid = 1
    for op in ops:
        if op.get("op") != "SUB":
            continue
        ref = op.get("ref") or ""
        hyp = op.get("hyp") or ""
        # check letter-level confusions (MVP)
        if "ص" in ref and "س" in hyp:
            conf, sev = "ص↔س", "high"
            flags.append({
                "id": f"f{fid}",
                "type": "MAKHRAJ_CONFUSION",
                "word": ref,
                "confusion": conf,
                "severity": sev,
                "note": f"Detected {ref} pronounced as {hyp}"
            })
            fid += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(flags, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
