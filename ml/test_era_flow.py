"""End-to-end check of era selection over HTTP.

    ml/.venv/Scripts/python -m ml.test_era_flow [--port 8000]

Covers the three things that can go wrong in a value-agreement vote: that the
options are offered at all, that a DISAGREEMENT blocks the start (rather than
silently handing someone the wrong player pool and engine), and that agreeing
actually locks the era in and lets the match run.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def _call(url, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return {"status": "error", "message": f"HTTP {e.code}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    def post(p, d):
        return _call(base + p, d)

    def get(p):
        return _call(base + p)

    failures = []

    a = post("/api/create_game", {"name": "Alpha"})
    ta, code = a["token"], a["code"]
    tb = post("/api/join_game", {"code": code, "name": "Bravo"})["token"]

    lob = get(f"/api/state?token={ta}").get("lobby") or {}
    eras = lob.get("eras") or []
    print(f"eras offered ({len(eras)}):")
    for e in eras:
        print(f"   {e['id']:<12} {e['first']}-{e['last']}  "
              f"{e['players']:>3} draftable   {e['label']}")
    if len(eras) < 2:
        failures.append("no era options exposed to the lobby")

    print("\n1. teams pick DIFFERENT eras")
    post("/api/vote_era", {"token": ta, "era": "2023_2026"})
    post("/api/vote_era", {"token": tb, "era": "2008_2013"})
    r = post("/api/quick_match", {"token": ta})
    print(f"   start blocked: {r.get('message')!r}")
    if r.get("status") == "success":
        failures.append("a disagreement did NOT block the start")

    print("\n2. they agree")
    r = post("/api/vote_era", {"token": tb, "era": "2023_2026"})
    print(f"   agreed={r.get('agreed')}  era={r.get('era')}")
    if not r.get("agreed") or r.get("era") != "2023_2026":
        failures.append("agreement did not lock the era in")

    print("\n3. match starts")
    r = post("/api/quick_match", {"token": ta})
    st = get(f"/api/state?token={ta}")
    print(f"   quick_match={r.get('status')}  phase={st.get('phase')}  "
          f"era={st.get('era')}")
    if r.get("status") != "success":
        failures.append(f"quick_match failed: {r.get('message')}")
    if st.get("era") != "2023_2026":
        failures.append(f"wrong era in play: {st.get('era')}")

    print()
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        raise SystemExit(1)
    print("PASS: era selection works end to end")


if __name__ == "__main__":
    main()
