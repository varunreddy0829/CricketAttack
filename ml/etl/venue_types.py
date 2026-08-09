"""Classify each ground into a PITCH TYPE from data, per era.

    ml/.venv/Scripts/python -m ml.etl.venue_types --era 2014_2022

The pitch types in config/ground_configs.json are hand-authored, and checking
them against the data showed three of ten are wrong -- Chepauk and Eden Gardens
are typed as the league's two turning tracks and both measure BELOW average for
spin, while Motera is typed green and measures spin-friendly. A hand label can't
be validated; a measured one can.

Two things this has to get right:

RESIDUALISE. Raw "spin economy minus pace economy" at a ground is dominated by
WHO bowled there. Chepauk reads spin-friendly on raw numbers purely because CSK
played Jadeja and Ashwin there for a decade. Comparing each bowler to his OWN
overall economy cancels that, and it flips Chepauk from -0.66 (spin-friendly) to
+0.92 (pace-friendly).

DEDUPE. The same ground appears under several names ("Wankhede Stadium" and
"Wankhede Stadium, Mumbai"; three spellings of Chepauk). Left split, each half
is measured on half the data -- and the disagreement between two halves of the
SAME ground is a free estimate of the noise floor, which came out at +/-0.15-0.25
economy. Anything smaller than that is not a pitch characteristic.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from ml.etl import eras as E
from ml.etl.replay import iter_innings
from ml.runtime.venues import canonical_ground
from src.utils.compile_player_stats import KNOWN_SPINNERS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Explicit, because normalising by rule silently produced an EMPTY key for
# "Dubai International Cricket Stadium" (every token was a stopword) and left
# Rajiv Gandhi split in two.
ALIASES = {
    "Wankhede Stadium": "wankhede",
    "Wankhede Stadium, Mumbai": "wankhede",
    "Eden Gardens": "eden",
    "Eden Gardens, Kolkata": "eden",
    "M Chinnaswamy Stadium": "chinnaswamy",
    "M.Chinnaswamy Stadium": "chinnaswamy",
    "M Chinnaswamy Stadium, Bengaluru": "chinnaswamy",
    "Feroz Shah Kotla": "kotla",
    "Arun Jaitley Stadium": "kotla",              # renamed in 2019, same ground
    "Arun Jaitley Stadium, Delhi": "kotla",
    "MA Chidambaram Stadium": "chepauk",
    "MA Chidambaram Stadium, Chepauk": "chepauk",
    "MA Chidambaram Stadium, Chepauk, Chennai": "chepauk",
    "Rajiv Gandhi International Stadium": "uppal",
    "Rajiv Gandhi International Stadium, Uppal": "uppal",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad": "uppal",
    "Punjab Cricket Association IS Bindra Stadium": "mohali",
    "Punjab Cricket Association IS Bindra Stadium, Mohali": "mohali",
    "Punjab Cricket Association Stadium, Mohali": "mohali",
    "Maharashtra Cricket Association Stadium": "pune",
    "Maharashtra Cricket Association Stadium, Pune": "pune",
    "Sawai Mansingh Stadium": "jaipur",
    "Sawai Mansingh Stadium, Jaipur": "jaipur",
    "Sardar Patel Stadium, Motera": "ahmedabad",
    "Narendra Modi Stadium, Ahmedabad": "ahmedabad",
    "Dr DY Patil Sports Academy": "dy_patil",
    "Dr DY Patil Sports Academy, Mumbai": "dy_patil",
    "Brabourne Stadium": "brabourne",
    "Brabourne Stadium, Mumbai": "brabourne",
    "Dubai International Cricket Stadium": "dubai",
    "Sharjah Cricket Stadium": "sharjah",
    "Sheikh Zayed Stadium": "abu_dhabi",
    "Zayed Cricket Stadium, Abu Dhabi": "abu_dhabi",
    "Saurashtra Cricket Association Stadium": "rajkot",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium": "vizag",
    "Holkar Cricket Stadium": "indore",
    "Barabati Stadium": "cuttack",
    "JSCA International Stadium Complex": "ranchi",
    "Green Park": "kanpur",
    "Himachal Pradesh Cricket Association Stadium": "dharamsala",
    "Vidarbha Cricket Association Stadium, Jamtha": "nagpur",
    "Subrata Roy Sahara Stadium": "pune",
    "Shaheed Veer Narayan Singh International Stadium": "raipur",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium": "lucknow",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow": "lucknow",
    "Narendra Modi Stadium": "ahmedabad",
    "Buffalo Park": "other", "De Beers Diamond Oval": "other",
    "OUTsurance Oval": "other", "St George's Park": "other",
    "Kingsmead": "other", "SuperSport Park": "other", "Newlands": "other",
    "New Wanderers Stadium": "other",
}

MIN_BALLS = 3000          # below this a ground's profile is mostly noise
MIN_TYPE_BALLS = 400      # per bowling type, for the residualised edge


def _key(venue: str) -> str:
    """ONE naming authority: ml/runtime/venues.py, the same map live play uses.

    An earlier version normalised by rule here instead, which produced its own
    ids ("kotla", "eden") that did not match the canonical ones the server looks
    up at ball time ("arun_jaitley", "eden_gardens") -- a profile written under
    one name and read under another is a silent fallback to league-neutral.
    """
    return canonical_ground(venue) or ALIASES.get(
        venue, venue.split(",")[0].strip().lower().replace(" ", "_"))


def collect(era: E.Era) -> dict:
    st = defaultdict(lambda: defaultdict(float))
    tot = defaultdict(lambda: [0, 0])
    at = defaultdict(lambda: [0, 0])

    for inn in iter_innings():
        if not era.covers(inn.season):
            continue
        for b in inn.balls:
            if not b.is_legal:
                continue
            v = _key(b.venue)
            d = st[v]
            d["balls"] += 1
            d["runs"] += b.runs_total
            d["dots"] += (b.outcome == "0")
            d["w"] += (b.outcome == "Out")
            d["bdry_runs"] += (4 if b.outcome == "4" else 6 if b.outcome == "6" else 0)
            if b.phase == "pp":
                d["pp_r"] += b.runs_total; d["pp_b"] += 1
            if b.phase == "death":
                d["de_r"] += b.runs_total; d["de_b"] += 1
            t = tot[b.bowler]; t[0] += b.runs_total; t[1] += 1
            a = at[(v, b.bowler)]; a[0] += b.runs_total; a[1] += 1

    # residualised spin/pace edge: each bowler against his OWN overall economy
    res = defaultdict(lambda: {"spin": [0.0, 0], "pace": [0.0, 0]})
    for (v, bw), (r, bl) in at.items():
        if bl < 60:
            continue
        tr, tb = tot[bw]
        if tb < 300:
            continue
        k = "spin" if bw in KNOWN_SPINNERS else "pace"
        c = res[v][k]
        c[0] += (6 * r / bl - 6 * tr / tb) * bl
        c[1] += bl

    out = {}
    for v, d in st.items():
        if d["balls"] < MIN_BALLS or v == "other":
            continue
        sp, pc = res[v]["spin"], res[v]["pace"]
        out[v] = {
            "balls": int(d["balls"]),
            "rpb": d["runs"] / d["balls"],
            "wpb": d["w"] / d["balls"],
            "dot": d["dots"] / d["balls"],
            "bdry_share": d["bdry_runs"] / d["runs"],
            "pp": d["pp_r"] / max(1, d["pp_b"]),
            "death": d["de_r"] / max(1, d["de_b"]),
            "spin_edge": (sp[0] / sp[1]) if sp[1] >= MIN_TYPE_BALLS else None,
            "pace_edge": (pc[0] / pc[1]) if pc[1] >= MIN_TYPE_BALLS else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--era", default="2014_2022")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    era = E.get(args.era)
    prof = collect(era)

    tb = sum(v["balls"] for v in prof.values())
    L = {k: sum(v[k] * v["balls"] for v in prof.values()) / tb
         for k in ("rpb", "wpb", "dot", "bdry_share", "pp", "death")}

    print(f"\n{era.id}: {len(prof)} grounds, {tb} balls")
    print(f"league  rpb {L['rpb']:.3f}  wpb {L['wpb']:.4f}  dot {100*L['dot']:.1f}%  "
          f"boundary-share {100*L['bdry_share']:.1f}%\n")

    hdr = (f"{'ground':<13}{'balls':>7}{'rpb':>7}{'bdry%':>7}{'wpb':>8}"
           f"{'dot%':>7}{'pp':>7}{'death':>7}{'spin':>7}{'pace':>7}{'s-p':>7}")
    print(hdr); print("-" * len(hdr))
    for v, x in sorted(prof.items(), key=lambda kv: -kv[1]["rpb"]):
        se, pe = x["spin_edge"], x["pace_edge"]
        sp = f"{se - pe:+.2f}" if (se is not None and pe is not None) else "--"
        print(f"{v:<13}{x['balls']:>7}{x['rpb']:>7.3f}{100*x['bdry_share']:>7.1f}"
              f"{x['wpb']:>8.4f}{100*x['dot']:>7.1f}{x['pp']:>7.3f}{x['death']:>7.3f}"
              f"{('--' if se is None else f'{se:+.2f}'):>7}"
              f"{('--' if pe is None else f'{pe:+.2f}'):>7}{sp:>7}")

    path = args.out or os.path.join(
        REPO_ROOT, "ml", "artifacts", "eras", era.id, "venue_profile.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"_league": L, "grounds": prof}, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
