#!/usr/bin/env python
"""Thin CLI: run generation only (router-expert system) over a JSONL of queries.

Delegates to :func:`src.inference.generation.main`. Run from the repo root:
    python scripts/generate.py --data_file data/triviaqa_test.jsonl --data-size 500
"""
from src.inference.generation import main

if __name__ == "__main__":
    main()
