#!/usr/bin/env python
"""Thin CLI: download + normalize the TriviaQA / ARC-C / IFEval eval sets.

Delegates to :func:`src.data.download_eval.main`. Run from the repo root:
    python scripts/download_eval_data.py
"""
from src.data.download_eval import main

if __name__ == "__main__":
    main()
