#!/usr/bin/env python
"""Thin CLI: train the 3-way router classification head.

Delegates to :func:`src.train.router.main`. Run from the repo root:
    python scripts/train_router.py --output_dir models/gemma270m-sft-router --bf16
"""
from src.train.router import main

if __name__ == "__main__":
    main()
