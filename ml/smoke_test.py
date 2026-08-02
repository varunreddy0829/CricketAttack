"""Drive a live server through a full over over HTTP.

    ml/.venv/Scripts/python -m ml.smoke_test [--port 8001]

Proves the whole path works in the real app, not just in the harness: two devices
join, quick match, toss, openers, and both halves of the over handshake. Point it
at :8000 for the classic engine or :8001 for the learned model -- it's the same
script, which is the point.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


def post(base, path, payload):
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "body": e.read().decode()[:300]}


def get(base, path):
    with urllib.request.urlopen(f"{base}{path}", timeout=10) as r:
        return json.loads(r.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="'localhost' costs ~2s per request in Python's urllib on Windows")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    a = post(base, "/api/create_game", {"name": "Alpha"})
    code, ta = a.get("code"), a.get("token")
    print(f"create_game -> code={code}")
    b = post(base, "/api/join_game", {"code": code, "name": "Bravo"})
    tb = b.get("token")
    print(f"join_game   -> ok={bool(tb)}")

    def ok(r):
        return r.get("status", r.get("error", "?"))

    print("quick_match ->", ok(post(base, "/api/quick_match", {"token": ta})))

    s = get(base, f"/api/state?token={ta}")
    m = s.get("match") or {}
    print(f"phase={s.get('phase')} stage={m.get('stage')}")

    # whoever won the toss decides
    for tok in (ta, tb):
        r = post(base, "/api/toss_choice", {"token": tok, "choice": "bat"})
        if r.get("status") == "success":
            print("toss_choice -> bat")
            break

    sa = get(base, f"/api/state?token={ta}")
    bat_tok = ta if (sa.get("match") or {}).get("i_am_batting") else tb
    bowl_tok = tb if bat_tok == ta else ta

    bench = (get(base, f"/api/state?token={bat_tok}").get("match") or {}).get("my_bench") or []
    names = [p["name"] for p in bench][:2]
    print("set_openers ->", ok(post(base, "/api/set_openers", {
        "token": bat_tok, "striker": names[0], "non_striker": names[1]})))

    # the over is SEQUENCED: the bowler must be locked in before the batting side
    # submits, so the batters never see who is bowling before choosing intent
    mb = get(base, f"/api/state?token={bowl_tok}").get("match") or {}
    bowlers = [p["name"] for p in (mb.get("my_bench") or []) if not p.get("disabled")]
    print("\nsubmit_over (bowling) ->", ok(post(base, "/api/submit_over", {
        "token": bowl_tok, "bowler_name": bowlers[0], "bowl_intent": 50,
        "bowl_role": "contain"})))

    print("submit_over (batting) ->", ok(post(base, "/api/submit_over", {
        "token": bat_tok, "striker_intent": 50, "non_striker_intent": 50,
        "striker_role": "rotate", "non_striker_role": "rotate"})))

    m = get(base, f"/api/state?token={bat_tok}").get("match") or {}
    print(f"\n--- after one over: {m.get('runs')}/{m.get('wickets')} "
          f"in {m.get('overs')} overs ---")
    for e in (m.get("this_over") or []):
        print(f"  {e.get('ball') or '':<6} {e.get('outcome') or '':<4} "
              f"{(e.get('text') or '')[:70]}")

    ok = m.get("balls", 0) > 0
    print(f"\n{'PASS' if ok else 'FAIL'}: {m.get('balls')} legal balls bowled")


if __name__ == "__main__":
    main()
