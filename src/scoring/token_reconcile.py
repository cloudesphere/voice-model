from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any


def strip_al(tok: str) -> str:
    return tok[2:] if tok.startswith("ال") and len(tok) > 2 else tok


def canon_wla(tok: str) -> str:
    # "ولا" -> "ول" to support merging patterns like: "ولا" + "الضالين" -> "ولضالين"
    if tok == "ولا":
        return "ول"
    if tok.startswith("ولا") and len(tok) > 3:
        return "ول" + tok[3:]
    return tok


def canon_token(tok: str) -> str:
    tok = (tok or "").strip()
    tok = canon_wla(tok)
    return tok


def eq_loose(a: str, b: str) -> bool:
    """
    Loose equality:
    - treat leading 'ال' as optional
    - normalize "ولا" => "ول"
    """
    a2 = canon_token(a)
    b2 = canon_token(b)

    if a2 == b2:
        return True
    if strip_al(a2) == b2:
        return True
    if a2 == strip_al(b2):
        return True
    if strip_al(a2) == strip_al(b2):
        return True
    return False


def merge_w_al(hyp_tokens: List[str]) -> List[str]:
    """
    Merge patterns:
      ['ولا', 'الضالين'] -> ['ولضالين']
      ['و',  'الضالين'] -> ['وضالين']
    """
    out: List[str] = []
    i = 0
    while i < len(hyp_tokens):
        t = canon_token(hyp_tokens[i])

        if i + 1 < len(hyp_tokens):
            n = canon_token(hyp_tokens[i + 1])
            if n.startswith("ال") and t in ("و", "ول", "ولا", "ف", "فل", "فلا"):
                out.append(canon_wla(t) + strip_al(n))
                i += 2
                continue

        out.append(t)
        i += 1

    return out


def choose_best_hyp_variant(
    ref_tokens: List[str],
    hyp_tokens: List[str],
    align_fn: Callable[[List[str], List[str], Callable[[str, str], bool]], Dict[str, Any]],
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """
    Deterministic: try original hypothesis vs merged-w-al hypothesis; pick lower WER.
    align_fn(ref, hyp, eq_pred) must return at least: wer, ins, dels.
    """
    ref = [canon_token(t) for t in ref_tokens]

    hyp1 = [canon_token(t) for t in hyp_tokens]
    res1 = align_fn(ref, hyp1, eq_loose)

    hyp2 = merge_w_al(hyp_tokens)
    res2 = align_fn(ref, hyp2, eq_loose)

    wer1 = float(res1.get("wer", 1.0))
    wer2 = float(res2.get("wer", 1.0))

    if wer2 < wer1:
        return hyp2, res2, {"variant": "merged_w_al"}
    if wer1 < wer2:
        return hyp1, res1, {"variant": "original"}

    # tie-break: fewer insertions+deletions
    e1 = float(res1.get("ins", 0)) + float(res1.get("dels", 0))
    e2 = float(res2.get("ins", 0)) + float(res2.get("dels", 0))
    if e2 < e1:
        return hyp2, res2, {"variant": "merged_w_al_tie"}

    return hyp1, res1, {"variant": "original_tie"}
