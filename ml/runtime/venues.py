"""One canonical name per real ground, shared by the stats builder and live play.

Cricsheet records the same physical ground under several strings -- "Wankhede
Stadium" vs "Wankhede Stadium, Mumbai", "M Chinnaswamy" vs "M.Chinnaswamy", the
Ekana ground under its full sponsor name. The game's own config/ground_configs.json
uses yet another short form ("PCA Stadium" for what Cricsheet calls "Punjab Cricket
Association ..."). Everything here keys off ONE canonical id per ground, matched by
substring, so `compute_venue_stats.py` (training-history side) and `server_ctx.py`
(live-play side) always agree on what "Wankhede" means.

Keep this list in sync with config/ground_configs.json's stadium names -- the test
in ml/tests/test_train_serve_parity.py checks every configured stadium resolves.
"""

from __future__ import annotations

# canonical id -> substrings that identify it in either Cricsheet or the game config.
# Order doesn't matter; each ground's substrings are distinct enough not to collide.
CANONICAL_GROUNDS: dict[str, list[str]] = {
    "chepauk":        ["Chidambaram", "Chepauk"],
    "eden_gardens":   ["Eden Gardens"],
    "pca_mohali":     ["Punjab Cricket Association", "PCA"],
    "narendra_modi":  ["Narendra Modi"],
    "wankhede":       ["Wankhede"],
    "chinnaswamy":    ["Chinnaswamy"],
    "sawai_mansingh": ["Sawai Mansingh"],
    "arun_jaitley":   ["Arun Jaitley"],
    "rajiv_gandhi":   ["Rajiv Gandhi"],
    "ekana":          ["Ekana"],
}


def canonical_ground(raw_name: str | None) -> str | None:
    """-> canonical id, or None if the name doesn't match any known ground."""
    if not raw_name:
        return None
    for key, needles in CANONICAL_GROUNDS.items():
        if any(n in raw_name for n in needles):
            return key
    return None
