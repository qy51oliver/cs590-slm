#!/usr/bin/env python
"""Thin CLI: evaluate the full router-expert system on a task (generate + score).

Delegates to :func:`src.eval.evaluate.main`. Run from the repo root:
    python scripts/evaluate.py --task arc-c --data-size 1000
"""
from src.eval.evaluate import main

if __name__ == "__main__":
    main()
