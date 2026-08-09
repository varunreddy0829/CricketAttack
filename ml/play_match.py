"""Drive a COMPLETE match through a running server, over HTTP.

    ml/.venv/Scripts/python -m ml.play_match --port 8001
    ml/.venv/Scripts/python -m ml.play_match --port 8000   # classic, same script

The one-over smoke test never touches the paths that actually break: a wicket and
the next-batter handshake, the free-hit window, the resume confirmation, the
innings break, a chase with a real target, and the finish. This walks the whole
`stage` machine:

    toss -> openers -> play -> (await_batter -> await_resume) -> (free_hit) -> ...
         -> innings break -> openers -> play -> ... -> finished

Exit code is non-zero if the match fails to complete, so it works as a check.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

MAX_STEPS = 400          # a 2-innings match is ~40 overs plus handshakes


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
        # a bare socket timeout / connection error is NOT an HTTPError -- without
        # this, one slow request hangs the whole driver with no diagnostic at all
        return {"status": "error", "message": f"{type(e).__name__}: {e}",
               "_transport_error": True}


class Driver:
    def __init__(self, base, verbose=False, req_timeout=8):
        self.base = base
        self.verbose = verbose
        self.req_timeout = req_timeout
        self.log = []

    def post(self, path, payload):
        return _req(f"{self.base}{path}", payload, timeout=self.req_timeout)

    def state(self, token):
        return _req(f"{self.base}/api/state?token={token}", timeout=self.req_timeout)

    # --- setup ------------------------------------------------------------
    def setup(self, era: str | None = None):
        a = self.post("/api/create_game", {"name": "Alpha"})
        self.ta, code = a["token"], a["code"]
        self.tb = self.post("/api/join_game", {"code": code, "name": "Bravo"})["token"]
        # Quick Match routes through the era screen too -- skipping the draft
        # doesn't mean skipping the choice of game. Request it, then agree on the
        # era, and the match starts itself.
        self.post("/api/quick_match", {"token": self.ta})
        for tok in (self.ta, self.tb):
            self.post("/api/vote_era", {"token": tok, "era": era or "all_time"})
        for tok in (self.ta, self.tb):
            if self.post("/api/toss_choice", {"token": tok, "choice": "bat"}).get(
                    "status") == "success":
                break
        return code

    def roles(self):
        """-> (batting token, bowling token) for the CURRENT innings."""
        m = self.state(self.ta).get("match") or {}
        return (self.ta, self.tb) if m.get("i_am_batting") else (self.tb, self.ta)

    # --- the stage machine ------------------------------------------------
    def set_openers(self, bat_tok):
        m = self.state(bat_tok).get("match") or {}
        avail = [p["name"] for p in (m.get("my_bench") or [])]
        if len(avail) < 2:
            return False
        return self.post("/api/set_openers", {
            "token": bat_tok, "striker": avail[0], "non_striker": avail[1],
        }).get("status") == "success"

    def bowl_and_bat(self, bat_tok, bowl_tok):
        """The over handshake -- bowling locks in first, by design."""
        import time
        mb = self.state(bowl_tok).get("match") or {}
        avail = [p["name"] for p in (mb.get("my_bench") or []) if not p.get("disabled")]
        if not avail:
            return "no_bowler"

        t0 = time.time()
        r = self.post("/api/submit_over", {
            "token": bowl_tok, "bowler_name": avail[0], "bowl_intent": 50,
            "bowl_role": "contain"})
        if self.verbose:
            print(f"    [bowl submit -> {r.get('status')} in {time.time()-t0:.1f}s]",
                  flush=True)
        if r.get("status") != "success":
            return f"bowl_rejected: {r.get('message')}"

        t0 = time.time()
        r = self.post("/api/submit_over", {
            "token": bat_tok, "striker_intent": 50, "non_striker_intent": 50,
            "striker_role": "rotate", "non_striker_role": "rotate"})
        dt = time.time() - t0
        if self.verbose or dt > 3:
            print(f"    [bat submit  -> {r.get('status')} in {dt:.1f}s "
                  f"-- THIS BALL RESOLUTION IS WHERE THE MODEL RUNS]", flush=True)
        if r.get("status") != "success":
            return f"bat_rejected: {r.get('message')}"
        return "ok"

    def next_batter(self, bat_tok):
        m = self.state(bat_tok).get("match") or {}
        avail = [p["name"] for p in (m.get("my_bench") or [])]
        for name in avail:
            r = self.post("/api/set_next_batter",
                          {"token": bat_tok, "batter_name": name})
            if r.get("status") == "success":
                return True
        return False

    def both(self, path):
        """Both sides confirm. Each side's fields are ignored for the other role,
        so one payload covers both."""
        payload = {"striker_intent": 50, "non_striker_intent": 50,
                   "striker_role": "rotate", "non_striker_role": "rotate",
                   "bowl_intent": 50, "bowl_role": "contain"}
        out = []
        for tok in (self.ta, self.tb):
            out.append(self.post(path, {"token": tok, **payload}).get("status"))
        return out

    # --- main loop --------------------------------------------------------
    def play(self, max_steps=MAX_STEPS, trace=False, era=None):
        code = self.setup(era)
        seen_innings = set()
        events = {"overs": 0, "wickets": 0, "free_hits": 0, "resumes": 0}
        stuck = {"key": None, "n": 0}

        for step in range(max_steps):
            bat_tok, bowl_tok = self.roles()
            s = self.state(bat_tok)
            phase = s.get("phase")
            m = s.get("match") or {}
            stage = m.get("stage")

            if trace:
                print(f"  step {step:>3}  stage={stage:<14} "
                      f"inn={m.get('innings')} {m.get('runs')}/{m.get('wickets')} "
                      f"ov={m.get('overs')}", flush=True)

            # a stage that repeats with no change in the score means the driver
            # can't advance it -- fail loudly instead of burning through max_steps
            key = (stage, m.get("innings"), m.get("runs"), m.get("wickets"),
                   m.get("balls"))
            if key == stuck["key"]:
                stuck["n"] += 1
                if stuck["n"] >= 4:
                    return False, events, {**m, "failure": f"stuck in stage {stage!r}"}
            else:
                stuck["key"], stuck["n"] = key, 0

            if phase == "finished" or stage == "done" or m.get("result"):
                return True, events, m

            innings = m.get("innings")
            if innings not in seen_innings:
                seen_innings.add(innings)
                if self.verbose:
                    print(f"\n--- innings {innings} "
                          f"(target {m.get('target') or '-'}) ---")

            if stage == "openers":
                if not self.set_openers(bat_tok):
                    return False, events, m
            elif stage == "play":
                r = self.bowl_and_bat(bat_tok, bowl_tok)
                if r != "ok":
                    return False, events, {**m, "failure": r}
                events["overs"] += 1
                if self.verbose:
                    m2 = (self.state(bat_tok).get("match") or {})
                    print(f"  ov {m2.get('overs'):>4}  {m2.get('runs')}/{m2.get('wickets')}")
            elif stage == "await_batter":
                events["wickets"] += 1
                if not self.next_batter(bat_tok):
                    return False, events, m
            elif stage == "await_resume":
                events["resumes"] += 1
                self.both("/api/ready_resume")
            elif stage == "free_hit":
                events["free_hits"] += 1
                self.both("/api/free_hit")
            elif stage == "toss":
                for tok in (self.ta, self.tb):
                    self.post("/api/toss_choice", {"token": tok, "choice": "bat"})
            else:
                return False, events, {**m, "failure": f"unknown stage {stage!r}"}

        return False, events, {"failure": "hit MAX_STEPS without finishing"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="'localhost' costs ~2s per request in Python's urllib on "
                         "Windows (IPv6-then-fallback) -- use the literal IP")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--trace", action="store_true", help="print every stage step")
    ap.add_argument("--max-steps", type=int, default=MAX_STEPS)
    ap.add_argument("--era", default=None, help="play in a specific era")
    args = ap.parse_args()
    base = f"http://{args.host}:{args.port}"

    failures = 0
    for i in range(args.matches):
        print(f"--- starting match {i + 1}/{args.matches} ---", flush=True)
        d = Driver(base, verbose=args.verbose)
        ok, ev, m = d.play(max_steps=args.max_steps, trace=args.trace, era=args.era)
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] match {i + 1}: {ev['overs']} overs, {ev['wickets']} wickets, "
              f"{ev['free_hits']} free hits, {ev['resumes']} resumes", flush=True)
        if ok:
            print(f"        result: {m.get('result') or '(none)'}", flush=True)
            for inn in (m.get("completed_innings") or []):
                print(f"        {inn.get('team', '?')}: {inn.get('runs')}/"
                      f"{inn.get('wickets')} ({inn.get('overs')} ov)", flush=True)
        else:
            failures += 1
            print(f"        FAILURE: {m.get('failure') or 'unknown'}", flush=True)
            print(f"        stage={m.get('stage')} innings={m.get('innings')} "
                  f"score={m.get('runs')}/{m.get('wickets')}", flush=True)

    print(f"\n{args.matches - failures}/{args.matches} matches completed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
