"""All-time vs modern-era calibration targets.

    ml/.venv/Scripts/python -m ml.harness.era_targets

Which one the simulator should aim at is a game-design call, not a statistical one.
Aiming at the all-time average produces cricket that stopped being played around
2016; aiming at 2023-26 produces the game people currently watch.
"""

from __future__ import annotations

from ml.harness.run_baseline import real_reference

KEYS = ("n_innings", "innings_mean", "innings_sd", "allout_rate",
        "rr_powerplay", "rr_middle", "rr_death", "dot_pct", "four_pct",
        "six_pct", "out_pct", "pct_over_200", "pct_under_120",
        "wides_per_innings")


def main() -> None:
    alltime, _ = real_reference()
    modern, _ = real_reference(season_min=2023)

    print("{:<20}{:>10}{:>10}{:>9}".format("metric", "all-time", "2023-26", "shift"))
    print("-" * 49)
    for k in KEYS:
        a, m = alltime[k], modern[k]
        print("{:<20}{:>10.2f}{:>10.2f}{:>+9.2f}".format(k, a, m, m - a))


if __name__ == "__main__":
    main()
