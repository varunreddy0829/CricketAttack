"""Play a full match in EVERY era, end to end over HTTP.

    ml/.venv/Scripts/python -m ml.play_all_eras

Each era has its own player pool, ball engine, calibration and league
baselines, so a wiring bug shows up as a crash or an obviously wrong score in
exactly one of them. All-time is included -- it runs the classic engine.
"""
from __future__ import annotations
import argparse
from ml.play_match import Driver, _req

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    a = _req(f"{base}/api/create_game", {"name": "probe"})
    _req(f"{base}/api/join_game", {"code": a["code"], "name": "probe2"})
    eras = (_req(f"{base}/api/state?token={a['token']}").get("lobby") or {}).get("eras", [])
    if not eras:
        raise SystemExit("server exposed no eras")

    fails = 0
    for e in eras:
        d = Driver(base)
        ok, ev, m = d.play(era=e["id"])
        fails += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {e['id']:<12} "
              f"{ev['overs']:>2} overs {ev['wickets']:>2} wkts  ->  "
              f"{m.get('result') or m.get('failure')}")
    print(f"\n{len(eras) - fails}/{len(eras)} eras completed a full match")
    raise SystemExit(1 if fails else 0)

if __name__ == "__main__":
    main()
