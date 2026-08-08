"""Era definitions, loaded once from config/eras.json.

Shared by everything era-scoped: the per-era ETL, the per-era model training and
calibration, the OVR derivation, and the server's per-era pool and engine
registries. One source of truth so a boundary can never be defined twice.
"""

from __future__ import annotations

import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ERAS_PATH = os.path.join(REPO_ROOT, "config", "eras.json")

ALL_TIME = "all_time"


class Era:
    def __init__(self, d: dict):
        self.id = d["id"]
        self.label = d["label"]
        self.tagline = d.get("tagline", "")
        self.first, self.last = d["seasons"]
        self.engine = d.get("engine", "model")
        self.min_bat_balls = d.get("min_bat_balls", 100)
        self.min_bowl_balls = d.get("min_bowl_balls", 200)

    @property
    def is_all_time(self) -> bool:
        return self.id == ALL_TIME

    @property
    def uses_model(self) -> bool:
        return self.engine == "model"

    def covers(self, season: int) -> bool:
        return self.first <= season <= self.last

    def __repr__(self) -> str:
        return f"Era({self.id}, {self.first}-{self.last}, {self.engine})"


def _load() -> dict:
    with open(ERAS_PATH, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {e["id"]: Era(e) for e in raw["eras"]}


ERAS: dict[str, Era] = _load()

# the eras with their own trained model, in chronological order
MODEL_ERAS = [e for e in ERAS.values() if e.uses_model]


def get(era_id: str) -> Era:
    if era_id not in ERAS:
        raise KeyError(f"unknown era {era_id!r}; known: {sorted(ERAS)}")
    return ERAS[era_id]


def era_dir(era_id: str, root: str) -> str:
    return os.path.join(root, "eras", era_id)


if __name__ == "__main__":
    print(f"{'id':<12}{'seasons':>12}{'engine':>9}  {'label'}")
    print("-" * 62)
    for e in ERAS.values():
        print(f"{e.id:<12}{f'{e.first}-{e.last}':>12}{e.engine:>9}  {e.label}")
