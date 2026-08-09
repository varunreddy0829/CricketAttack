"""Drive a COMPLETE auction through a running server, over HTTP.

    ml/.venv/Scripts/python -m ml.play_auction --port 8002 --era 2023_2026

Verification item 4 of the era plan asked whether the auction still fills in a
smaller era pool. Calling generate_draft_pool directly answers a weaker question
than it looks: it says the SETS build, not that two teams can actually bid their
way through them and come out with legal squads. The parts that only a live run
exercises are the purse arithmetic, the background going-once/twice timer, the
end-of-auction auto-fill top-up, and the 15-21 / >=1 keeper squad validation --
and auto-fill is exactly the mechanism a short era set leans on.

Bidding strategy here is deliberately dumb: team A bids up to a per-lot cap, B
folds. That is not a realistic auction, but it is the one that stresses the thing
under test -- one squad gets thin and has to be topped up by auto-fill.

Exit code is non-zero if either squad ends illegal, so it works as a check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

MAX_STEPS = 4000
MIN_SQUAD, MAX_SQUAD = 15, 21


def _req(url, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method="POST" if data else "GET",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return {"status": "error", "message": json.loads(e.read()).get("message", "")}
        except Exception:
            return {"status": "error", "message": f"HTTP {e.code}"}
    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


class AuctionDriver:
    def __init__(self, base, cap=12.0, verbose=False):
        self.base = base
        self.cap = cap                 # team A's per-lot ceiling, in Cr
        self.verbose = verbose

    def post(self, path, payload):
        return _req(f"{self.base}{path}", payload)

    def state(self, token):
        return _req(f"{self.base}/api/state?token={token}")

    def setup(self, era):
        a = self.post("/api/create_game", {"name": "Alpha"})
        if a.get("status") == "error":
            raise SystemExit(f"create_game failed: {a.get('message')}")
        self.ta, code = a["token"], a["code"]
        b = self.post("/api/join_game", {"code": code, "name": "Bravo"})
        self.tb = b["token"]
        # Ready first, then the era screen, then home grounds -- era selection
        # is its own step now rather than a strip inside the lobby.
        for tok in (self.ta, self.tb):
            self.post("/api/start_auction", {"token": tok})
        if era:
            for tok in (self.ta, self.tb):
                self.post("/api/vote_era", {"token": tok, "era": era})
        self.pick_grounds()
        return code

    def pick_grounds(self):
        """Home grounds are chosen BEFORE the auction now, and each pick is
        hidden -- a ground someone else holds reports only `taken`."""
        for tok in (self.ta, self.tb):
            s = self.state(tok)
            if s.get("phase") != "grounds":
                return
            gr = s.get("grounds") or {}
            free = [g["id"] for g in (gr.get("stadiums") or []) if not g.get("taken")]
            if not free:
                return
            self.post("/api/claim_ground", {"token": tok, "ground_id": free[0]})
            self.post("/api/lock_ground", {"token": tok})

    def run(self, era, max_steps=MAX_STEPS):
        self.setup(era)
        sold = 0
        seen_lot = None
        stalls = 0

        for _ in range(max_steps):
            s = self.state(self.ta)
            if s.get("phase") != "auction":
                break
            a = s.get("auction") or {}
            stage = a.get("stage")

            if stage == "done":
                break
            if stage == "preview":
                for tok in (self.ta, self.tb):
                    self.post("/api/auction_ready", {"token": tok})
                continue
            if stage == "resolved":
                for tok in (self.ta, self.tb):
                    self.post("/api/auction_ready", {"token": tok})
                time.sleep(0.05)
                continue
            if stage == "bidding":
                cur = a.get("current") or {}
                lot = cur.get("name")
                bid = a.get("current_bid") or 0
                leader = a.get("active_bidder")
                me = a.get("you_role")

                if lot != seen_lot:
                    seen_lot, stalls = lot, 0
                    sold += 1
                else:
                    stalls += 1

                # A bids while under its cap and not already leading; B always
                # folds. Both sides then pass, so the going-once/twice timer --
                # which runs on the server's own thread -- resolves the lot.
                if bid <= self.cap and leader != me:
                    self.post("/api/bid", {"token": self.ta, "amount": bid})
                self.post("/api/pull_out", {"token": self.tb})
                self.post("/api/pull_out", {"token": self.ta})
                time.sleep(0.05)
                continue

            time.sleep(0.05)

        return self.report(), sold

    def report(self):
        out = {}
        for tok in (self.ta, self.tb):
            s = self.state(tok)
            a = s.get("auction") or {}
            sq = a.get("my_squad") or {}
            roster = sq.get("roster") or []
            out[sq.get("name", tok[:4])] = {
                "n": sq.get("count", len(roster)),
                "keepers": sq.get("wk", 0),
                "overseas": sq.get("os", 0),
                "budget": sq.get("budget"),
                "phase": s.get("phase"),
                "stage": a.get("stage"),
            }
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8002)
    ap.add_argument("--era", default=None)
    ap.add_argument("--cap", type=float, default=12.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    d = AuctionDriver(f"http://{args.host}:{args.port}", cap=args.cap,
                      verbose=args.verbose)
    t0 = time.time()
    res, lots = d.run(args.era)

    failures = []
    print(f"era {args.era or 'all_time':<12} {lots} lots  ({time.time() - t0:.0f}s)")
    for role, r in res.items():
        legal = MIN_SQUAD <= r["n"] <= MAX_SQUAD and r["keepers"] >= 1
        print(f"  {role:<8} {r['n']:>2} players, {r['keepers']} keeper(s), "
              f"{r['overseas']} overseas, budget {r['budget']}  "
              f"[{r['stage']}]  {'ok' if legal else 'ILLEGAL'}")
        if not legal:
            failures.append(f"{role}: {r['n']} players / {r['keepers']} keepers "
                            f"-- needs {MIN_SQUAD}-{MAX_SQUAD} and >=1 keeper")

    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        sys.exit(1)
    print("  auction PASSED")


if __name__ == "__main__":
    main()
