"""Cricsheet extraction. `replay.py` is the single source of truth for eligibility
filtering and derived per-ball match state — the reference statistics and the feature
table are both built from it, so their filters can never drift apart."""
