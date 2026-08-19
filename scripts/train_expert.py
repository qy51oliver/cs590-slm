#!/usr/bin/env python
"""Thin CLI: prompt-masked SFT for a single expert.

Delegates to :func:`src.train.expert.main`. Run from the repo root:
    python scripts/train_expert.py --train_file data/v3/v3_reasoning_train.jsonl \\
        --model google/gemma-3-270m --output_dir models/gemma270m-sft-reasoning \\
        --max_length 512 --bf16
"""
from src.train.expert import main

if __name__ == "__main__":
    main()
