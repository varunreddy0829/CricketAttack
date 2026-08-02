"""Model fitting. `backbone.py` is pure numpy (the wide half); `deep_head.py`
adds the torch residual (the deep half). Offline only -- nothing here is imported
by the runtime, which reads exported .npz arrays."""
