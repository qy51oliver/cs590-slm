#!/usr/bin/env python
"""Thin CLI: build the three expert SFT corpora (unified JSONL).

Delegates to :func:`src.data.prepare_sft.main`. Run from the repo root:
    python scripts/prepare_sft_data.py --if-filter-constraints
"""
from src.data.prepare_sft import main

if __name__ == "__main__":
    main()
