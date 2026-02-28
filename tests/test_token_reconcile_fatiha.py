"""
Deterministic unit tests for token reconciliation (optional الـ, ولا+ال merge).
No CLI or ASR; calls levenshtein_ops and choose_best_hyp_variant directly.
"""
from __future__ import annotations

import sys
import os

# Allow importing src from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scoring.word_scorer_cli import levenshtein_ops
from src.scoring.token_reconcile import eq_loose, choose_best_hyp_variant


def test_sirat_al_match():
    """ref صراط vs hyp الصراط: optional leading ال → MATCH, no SUB/INS/DEL."""
    res = levenshtein_ops(["صراط"], ["الصراط"], eq_pred=eq_loose)
    assert res["subs"] == 0, f"expected 0 subs, got {res['subs']}"
    assert res["ins"] == 0, f"expected 0 ins, got {res['ins']}"
    assert res["dels"] == 0, f"expected 0 dels, got {res['dels']}"


def test_waladhalin_merged_match():
    """ref ولضالين vs hyp ولا + الضالين: merged variant → MATCH, no SUB/INS/DEL."""
    def align_fn(ref_toks, hyp_toks, eq_pred):
        return levenshtein_ops(ref_toks, hyp_toks, eq_pred=eq_pred)

    _hyp_best, align_res, _variant_meta = choose_best_hyp_variant(
        ["ولضالين"],
        ["ولا", "الضالين"],
        align_fn=align_fn,
    )
    assert align_res["subs"] == 0, f"expected 0 subs, got {align_res['subs']}"
    assert align_res["ins"] == 0, f"expected 0 ins, got {align_res['ins']}"
    assert align_res["dels"] == 0, f"expected 0 dels, got {align_res['dels']}"


if __name__ == "__main__":
    test_sirat_al_match()
    test_waladhalin_merged_match()
    print("All tests passed.")
