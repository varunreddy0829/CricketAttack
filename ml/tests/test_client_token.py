"""Every client POST that needs a session token must actually send one.

This is a STATIC check on src/public/, because the bug it guards against is
invisible to every other test we have: the API suites build their own requests
and always pass a token explicitly, so they pass while the real UI fails.

The bug: renderEraScreen called

    Net.post('/api/vote_era', { era: ... })

with no token, and Net.post did not add one. The server's _game_by_token returned
None and answered "Game not found or session expired." -- which reads like a dead
session rather than a missing field, so it sent the hunt in the wrong direction.
Clicking an era simply did nothing.
"""

from __future__ import annotations

import os
import re
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PUBLIC = os.path.join(REPO, "src", "public")

# The only endpoints that legitimately run without a token: they MINT one.
# Verified against src/server.py -- none of these read an incoming token.
NO_TOKEN_NEEDED = {
    "/api/create_game", "/api/join_game",
    "/api/create_tournament", "/api/join_tournament",
}


class TestClientSendsToken(unittest.TestCase):
    def test_net_post_defaults_the_token(self):
        """Net.post must attach the stored token when the caller omits it."""
        with open(os.path.join(PUBLIC, "net.js"), "r", encoding="utf-8") as fh:
            js = fh.read()
        body = js[js.index("async function post("):js.index("async function get(")]
        self.assertIn(
            "getToken()", body,
            "Net.post no longer reads the stored token -- any call site that "
            "omits `token` will fail as 'Game not found or session expired'")
        self.assertRegex(
            body, r"payload\.token\s*=\s*tok",
            "Net.post reads the token but never puts it on the payload")

    def test_every_endpoint_is_reachable_with_a_token(self):
        """No call site posts to a token-requiring endpoint with an explicit
        `token: undefined`/null, which would defeat the default above."""
        with open(os.path.join(PUBLIC, "app.js"), "r", encoding="utf-8") as fh:
            js = fh.read()
        bad = []
        for m in re.finditer(r"Net\.post\(\s*'([^']+)'\s*,\s*\{([^}]*)\}", js):
            path, args = m.group(1), m.group(2)
            if path in NO_TOKEN_NEEDED:
                continue
            if re.search(r"token\s*:\s*(undefined|null)", args):
                line = js[:m.start()].count("\n") + 1
                bad.append(f"app.js:{line} posts to {path} with an empty token")
        self.assertEqual(bad, [], "\n".join(bad))


if __name__ == "__main__":
    unittest.main()
