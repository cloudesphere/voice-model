from __future__ import annotations
import argparse, json, os, re
from typing import List, Dict, Any

def normalize_ar(s: str) -> str:
    import re
    if not s:
        return ""
    s = s.strip().lower()

    # Remove harakat and tatweel
    s = re.sub(r"[ًٌٍَُِّْـ]", "", s)

    # Remove punctuation (Arabic + English) safely
    punct = "،؛؟.,!?:;()[]{}-"
    trans = str.maketrans({c: " " for c in punct})
    s = s.translate(trans)

    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


SEGMENT_KEYWORDS = {
    # MVP: seg_full_0001 = Al-Fatiha core keywords (harakat removed)
    "seg_full_0001": [
        "الحمد","رب","العالمين","الرحمن","الرحيم","مالك","يوم","الدين",
        "اياك","نعبد","نستعين","اهدنا","الصراط","المستقيم","غير","المغضوب","الضالين"
    ]
}

def keyword_hits(text: str, segment_id: str) -> int:
    keys = SEGMENT_KEYWORDS.get(segment_id) or []
    if not keys:
        return 0
    t = normalize_ar(text)
    return sum(1 for k in keys if k in t)

def enforce_segment_guard(hyp_text: str, segment_id: str, min_hits: int = 2) -> None:
    # If we don't have a keyword list for segment, do nothing.
    if segment_id not in SEGMENT_KEYWORDS:
        return
    hits = keyword_hits(hyp_text, segment_id)
    if hits < min_hits:
        raise ValueError(
            f"AUDIO_MISMATCH_SEGMENT: ASR does not look like {segment_id}. "
            f"keyword_hits={hits} < {min_hits}"
        )

def load_ref_words(alignment_path: str) -> List[str]:
    aln = json.load(open(alignment_path, "r", encoding="utf-8"))

    def extract_from_list(lst):
        out=[]
        for w in lst:
            if isinstance(w, dict):
                t = w.get("text") or w.get("word") or w.get("token")
                if t:
                    out.append(str(t))
            elif isinstance(w, str):
                out.append(w)
        return out

    # Candidate locations (seen in various alignment formats)
    candidates = []

    # direct lists
    for k in ("word_timestamps", "words"):
        v = aln.get(k)
        if isinstance(v, list) and v:
            candidates.append(v)

    # nested objects
    for k in ("word_timestamps", "words", "alignment"):
        v = aln.get(k)
        if isinstance(v, dict):
            for kk in ("words","word_timestamps","items"):
                vv = v.get(kk)
                if isinstance(vv, list) and vv:
                    candidates.append(vv)

    # segments -> words
    segs = aln.get("segments")
    if isinstance(segs, list):
        for seg in segs:
            if isinstance(seg, dict):
                vv = seg.get("words") or seg.get("word_timestamps")
                if isinstance(vv, list) and vv:
                    candidates.append(vv)

    # pick first non-empty extraction
    for lst in candidates:
        out = extract_from_list(lst)
        if out:
            return out

    # Fallback: if no words found, do NOT use char-level tokens as words
    return []

def whisper_transcribe(wav_path: str) -> str:
    # faster-whisper
    from faster_whisper import WhisperModel
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(wav_path, language="ar", beam_size=5)
    text = " ".join(seg.text for seg in segments)
    return text


def trim_ref_to_hyp(ref: list[str], hyp: list[str], min_anchor_hits: int = 1, min_matches: int = 3) -> tuple[list[str], dict]:
    """
    Partial-aware scoring with start anchor:
    - Find the earliest reasonable start index in ref that matches hyp[0] (or first anchorable hyp word).
    - Then find last matched index in-order starting from that start.
    - Use ref[start:last+1] for scoring, and record skipped prefix.
    """
    meta = {
        "partial_mode": True,
        "ref_len_original": len(ref),
        "ref_len_used": len(ref),
        "matched_in_order": 0,
        "start_ref_index": 0,
        "last_ref_index": -1,
        "trim_applied": False,
        "skipped_ref_prefix": [],
    }
    if not ref or not hyp:
        meta["partial_mode"] = False
        return ref, meta

    # Choose an anchor word from hyp that exists in ref
    anchor = None
    for w in hyp[:6]:  # look at first few hyp words
        if w in ref:
            anchor = w
            break

    if anchor is None:
        meta["partial_mode"] = False
        return ref, meta

    # Start at earliest occurrence of anchor in ref
    start = ref.index(anchor)
    meta["start_ref_index"] = start
    meta["skipped_ref_prefix"] = ref[:start]

    # Now match hyp in-order starting from start
    i = start
    last = -1
    matches = 0
    for w in hyp:
        while i < len(ref) and ref[i] != w:
            i += 1
        if i < len(ref) and ref[i] == w:
            matches += 1
            last = i
            i += 1

    meta["matched_in_order"] = matches
    meta["last_ref_index"] = last

    if matches >= min_matches and last >= start:
        # Use a window roughly matching hyp length to avoid overly short ref (which inflates INS).
        win = min(len(ref) - start, max(3, len(hyp) + 2))
        end = max(last + 1, start + win)
        end = min(end, len(ref))
        trimmed = ref[start:end]
        meta["ref_len_used"] = len(trimmed)
        meta["trim_applied"] = True
        meta["trim_last_word"] = trimmed[-1] if trimmed else None
        return trimmed, meta

    # Fallback: if weak evidence, just use ref from start without trimming end
    trimmed = ref[start:]
    meta["ref_len_used"] = len(trimmed)
    meta["trim_applied"] = True
    meta["trim_last_word"] = trimmed[-1] if trimmed else None
    return trimmed, meta

    i = 0
    last = -1
    matches = 0
    for w in hyp:
        # advance ref pointer until match found
        while i < len(ref) and ref[i] != w:
            i += 1
        if i < len(ref) and ref[i] == w:
            matches += 1
            last = i
            i += 1  # continue after match

    meta["matched_in_order"] = matches
    meta["last_ref_index"] = last

    # Only apply trimming if we have enough evidence of prefix recitation
    if matches >= min_matches and last >= 0:
        trimmed = ref[: last + 1]
        meta["ref_len_used"] = len(trimmed)
        meta["trim_applied"] = True
        meta["trim_last_word"] = trimmed[-1] if trimmed else None
        return trimmed, meta

    meta["partial_mode"] = False
    meta["trim_applied"] = False
    return ref, meta

def levenshtein_ops(ref: List[str], hyp: List[str]) -> Dict[str, Any]:
    # DP alignment producing ops (MVP)
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    bt = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        dp[i][0] = i; bt[i][0] = ("DEL", i-1, None)
    for j in range(1, m+1):
        dp[0][j] = j; bt[0][j] = ("INS", None, j-1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost_sub = 0 if ref[i-1] == hyp[j-1] else 1
            cand = [
                (dp[i-1][j] + 1, ("DEL", i-1, None)),
                (dp[i][j-1] + 1, ("INS", None, j-1)),
                (dp[i-1][j-1] + cost_sub, ("EQ" if cost_sub==0 else "SUB", i-1, j-1)),
            ]
            dp[i][j], bt[i][j] = min(cand, key=lambda x: x[0])
    # backtrace
    i, j = n, m
    ops = []
    subs = dels = ins = matches = 0
    while i>0 or j>0:
        op, ri, hj = bt[i][j]
        if op == "DEL":
            ops.append({"ref": ref[ri], "hyp": None, "op": "DEL"})
            dels += 1
            i -= 1
        elif op == "INS":
            ops.append({"ref": None, "hyp": hyp[hj], "op": "INS"})
            ins += 1
            j -= 1
        else:
            if op == "EQ":
                matches += 1
                ops.append({"ref": ref[ri], "hyp": hyp[hj], "op": "MATCH"})
            else:
                subs += 1
                ops.append({"ref": ref[ri], "hyp": hyp[hj], "op": "SUB"})
            i -= 1; j -= 1
    ops.reverse()
    wer = (subs + dels + ins) / max(1, n)
    # Cap for downstream scoring (engine expects 0..1)
    wer = min(1.0, max(0.0, wer))
    if wer < 0.0:
        wer = 0.0
    if wer > 1.0:
        wer = 1.0
    return {"wer": round(wer, 4), "matches": matches, "subs": subs, "dels": dels, "ins": ins, "ops": ops}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user_wav", required=True)
    ap.add_argument("--segment_id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    aln_path = os.path.join("artifacts", "alignments", f"{args.segment_id}.alignment.json")
    ref_words = load_ref_words(aln_path)
    ref_norm = normalize_ar(" ".join(ref_words)).split()

    hyp_text = whisper_transcribe(args.user_wav)
    enforce_segment_guard(hyp_text, args.segment_id, min_hits=2)
    hyp_norm = normalize_ar(hyp_text).split()

    ref_used, partial_meta = trim_ref_to_hyp(ref_norm, hyp_norm, min_matches=3)
    out = levenshtein_ops(ref_used, hyp_norm)
    out["meta"] = partial_meta
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
