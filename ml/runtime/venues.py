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
# Renamed grounds list BOTH names: they are one physical pitch and splitting them
# halves the history behind its measured character. Delhi's ground was Feroz Shah
# Kotla until 2019 and Arun Jaitley after; Ahmedabad's was Sardar Patel (Motera)
# before the Narendra Modi rebuild.
CANONICAL_GROUNDS: dict[str, list[str]] = {
    "chepauk":        ["Chidambaram", "Chepauk"],
    "eden_gardens":   ["Eden Gardens"],
    "pca_mohali":     ["Punjab Cricket Association", "PCA"],
    "narendra_modi":  ["Narendra Modi", "Sardar Patel", "Motera"],
    "wankhede":       ["Wankhede"],
    "chinnaswamy":    ["Chinnaswamy"],
    "sawai_mansingh": ["Sawai Mansingh"],
    "arun_jaitley":   ["Arun Jaitley", "Feroz Shah Kotla"],
    "rajiv_gandhi":   ["Rajiv Gandhi", "Uppal"],
    "ekana":          ["Ekana"],
    # not pickable in-game, but they carry real IPL history and the model trains
    # on their balls, so they need stable ids too
    "dubai":          ["Dubai International"],
    "sharjah":        ["Sharjah"],
    "abu_dhabi":      ["Sheikh Zayed", "Zayed Cricket"],
    "brabourne":      ["Brabourne"],
    "dy_patil":       ["DY Patil"],
    "pune_mca":       ["Maharashtra Cricket Association", "Subrata Roy"],
}


def canonical_ground(raw_name: str | None) -> str | None:
    """-> canonical id, or None if the name doesn't match any known ground."""
    if not raw_name:
        return None
    for key, needles in CANONICAL_GROUNDS.items():
        if any(n in raw_name for n in needles):
            return key
    return None
