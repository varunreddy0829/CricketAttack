/* ============================================================
   net.js — talks to the authoritative server.
   Stores the player token, polls /api/state, and dispatches a
   'gamestate' event whenever the server's version changes.
   ============================================================ */

const Net = (() => {
    const TOKEN_KEY = 'ca_token';

    function getToken() { return localStorage.getItem(TOKEN_KEY); }
    function setToken(t) { localStorage.setItem(TOKEN_KEY, t); }
    function clearToken() { localStorage.removeItem(TOKEN_KEY); }

    async function post(path, body) {
        // Attach the session token unless the caller passed one (or is
        // create/join, which has no token yet and where getToken() is null).
        //
        // Every endpoint bar create/join identifies the caller by token, and
        // omitting it fails as "Game not found or session expired" -- which
        // reads like an expired session rather than a missing field. Defaulting
        // it here means a new call site cannot make that mistake.
        const payload = body || {};
        const tok = getToken();
        if (tok && payload.token === undefined) payload.token = tok;
        const res = await fetch(path, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.status === 'error') {
            const err = new Error(data.message || `Request failed (${res.status})`);
            err.data = data;   // extra fields (e.g. { full: true, roster: [...] }) survive on the error
            throw err;
        }
        return data;
    }

    async function get(path) {
        const res = await fetch(path);
        const data = await res.json().catch(() => ({}));
        if (!res.ok || data.status === 'error') {
            throw new Error(data.message || `Request failed (${res.status})`);
        }
        return data;
    }

    async function fetchState() {
        const token = getToken();
        if (!token) return null;
        const res = await fetch(`/api/state?token=${encodeURIComponent(token)}`);
        return res.json();
    }

    let lastVersion = -1;
    let timer = null;

    async function tick() {
        try {
            const state = await fetchState();
            if (state && state.version !== undefined && state.version !== lastVersion) {
                lastVersion = state.version;
                window.dispatchEvent(new CustomEvent('gamestate', { detail: state }));
            }
        } catch (e) {
            /* transient network hiccup — keep polling */
        }
    }

    function startPolling(intervalMs = 700) {
        if (timer) return;
        tick();
        timer = setInterval(tick, intervalMs);
    }

    function forceRefresh() { lastVersion = -1; return tick(); }

    return { getToken, setToken, clearToken, post, get, startPolling, forceRefresh };
})();
