"""Side-by-side report: simulated distributions against real IPL."""

from __future__ import annotations

from ml.harness.stats import ACCEPTANCE

_W = 22


def report(real: dict, sim: dict, *, title: str = "simulated") -> int:
    """Print the comparison table. Returns the number of metrics outside tolerance."""
    print()
    print(f"  {'metric':<{_W}} {'real':>9} {title:>9} {'delta':>9} {'tol':>8}   verdict")
    print("  " + "-" * (_W + 42))

    failed = 0
    for key, tol, unit in ACCEPTANCE:
        if key not in real or key not in sim:
            continue
        r, s = real[key], sim[key]
        d = s - r
        ok = abs(d) <= tol
        failed += 0 if ok else 1
        mark = "ok" if ok else "OFF"
        print(f"  {key:<{_W}} {r:>9.2f} {s:>9.2f} {d:>+9.2f} {'±' + f'{tol:g}':>8}   {mark}")

    print("  " + "-" * (_W + 42))
    print(f"  {failed}/{len(ACCEPTANCE)} metrics outside tolerance"
          f"   (real n={real.get('n_innings', 0)}, sim n={sim.get('n_innings', 0)})")

    # context-only metrics, no tolerance attached
    extra = ["innings_p10", "innings_p50", "innings_p90",
             "top_score_mean", "fifty_plus_per_innings", "one_pct"]
    print()
    print(f"  {'(context only)':<{_W}} {'real':>9} {title:>9}")
    for key in extra:
        if key in real and key in sim:
            print(f"  {key:<{_W}} {real[key]:>9.2f} {sim[key]:>9.2f}")
    print()
    return failed
