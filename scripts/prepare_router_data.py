#!/usr/bin/env python
"""Thin CLI: build the balanced 3-way router classification dataset.

Delegates to :func:`src.data.prepare_router.main`. Run from the repo root:
    python scripts/prepare_router_data.py
"""
from src.data.prepare_router import main

if __name__ == "__main__":
    main()
