# voice-model

**Tests:** From project root, run `python tests/test_token_reconcile_fatiha.py` or `pip install pytest && pytest tests/`.

Token reconciliation: optional الـ and merge wala+al patterns (e.g., ولا الضالين) improves WER deterministically. Leading "ال" is treated as optional in token matching (e.g., صراط ↔ الصراط), and split/merge cases like ref "ولضالين" vs hypothesis "ولا" + "الضالين" are handled by trying a merged hypothesis variant so alignment produces a match instead of spurious SUB/INS.
