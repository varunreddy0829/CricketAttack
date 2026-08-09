"""A TOURNAMENT must ask which era to play, just like a 1v1 does.

    ml/.venv/Scripts/python -m ml.test_tournament_era [--port 8002]

This exists because it silently did not. Tournaments render their own lobby
(renderTournamentLobby) and never called renderEraPick, and the tournament_lobby
payload carried no era fields at all -- so no picker could appear, nobody voted,
and _era_block_reason -- deliberately permissive when NOBODY has voted -- waved
the game straight through to grounds and then the auction on the all-time
default. Nothing errored. You just never got asked.

ml/test_era_flow.py covers the 1v1 path and passed throughout, because the gap
was only ever in the tournament branch.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

TEAMS = 4


def _call(url, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    def post(p, d):
        return _call(base + p, d)

    def get(t):
        return _call(f"{base}/api/state?token={t}")[1]

    failures = []

    _, a = post("/api/create_tournament", {"name": "T1", "teams": TEAMS})
    toks = [a["token"]]
    for n in range(2, TEAMS + 1):
        _, j = post("/api/join_tournament", {"code": a["code"], "name": f"T{n}"})
        toks.append(j["token"])

    tl = get(toks[0]).get("tournament_lobby") or {}
    eras = tl.get("eras") or []
    playable = [e["id"] for e in eras if not e.get("coming_soon")]
    print(f"1. tournament lobby exposes {len(eras)} eras, {len(playable)} playable")
    if not eras:
        failures.append("tournament_lobby carries no era options -- no picker can render")
    if len(playable) < 2:
        failures.append(f"need 2 playable eras, found {playable}")
        _report(failures)
        return
    first, second = playable[0], playable[1]

    print("\n2. teams pick DIFFERENT eras")
    post("/api/vote_era", {"token": toks[0], "era": first})
    post("/api/vote_era", {"token": toks[1], "era": second})
    tl = get(toks[0])["tournament_lobby"]
    print(f"   era_clash={tl.get('era_clash')}  voted={tl.get('voted_count')}/{tl.get('size')}")
    if not tl.get("era_clash"):
        failures.append("a genuine era clash was not reported to the lobby")
    code, body = post("/api/start_auction", {"token": toks[0]})
    print(f"   start_auction -> HTTP {code}: {body.get('message')!r}")
    if code == 200:
        failures.append("the auction started while teams disagreed on the era")

    print("\n3. everyone agrees")
    for t in toks:
        post("/api/vote_era", {"token": t, "era": first})
    tl = get(toks[0])["tournament_lobby"]
    print(f"   era_agreed={tl.get('era_agreed')}  clash={tl.get('era_clash')}")
    if tl.get("era_agreed") != first:
        failures.append(f"agreement did not lock the era in ({tl.get('era_agreed')})")

    print("\n4. through grounds into the auction")
    for t in toks:
        post("/api/start_auction", {"token": t})
    st = get(toks[0])
    print(f"   phase={st.get('phase')}  era={st.get('era')}")
    if st.get("phase") != "grounds":
        failures.append(f"expected the grounds phase first, got {st.get('phase')}")
    for t in toks:
        gr = get(t).get("grounds") or {}
        free = [g["id"] for g in (gr.get("stadiums") or []) if not g.get("taken")]
        if free:
            post("/api/claim_ground", {"token": t, "ground_id": free[0]})
            post("/api/lock_ground", {"token": t})
    st = get(toks[0])
    lots = len((st.get("auction") or {}).get("pool") or [])
    print(f"   phase={st.get('phase')}  era={st.get('era')}  "
          f"label={st.get('era_label')!r}  lots={lots}")
    if st.get("phase") != "auction":
        failures.append(f"the auction did not open ({st.get('phase')})")
    # THE assertion this file exists for: the chosen era is the one being played,
    # not the silent all-time default.
    if st.get("era") != first:
        failures.append(f"tournament is playing {st.get('era')}, not the agreed {first}")

    _report(failures)


def _report(failures):
    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("PASS: a tournament picks its era, and plays the one it picked")


if __name__ == "__main__":
    main()
