"""Router-expert SLM: task-specialized fine-tuning of Gemma-3-270M.

Package layout:
    src.data       dataset download + SFT/router data preparation
    src.train      supervised fine-tuning for experts and the router
    src.inference  the router-expert inference pipeline + generation helpers
    src.eval       evaluation, scoring, and ablation studies
"""

__version__ = "0.1.0"
