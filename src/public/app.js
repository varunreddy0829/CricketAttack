/* ============================================================
   app.js — screens, rendering, and match interactions.
   The server is authoritative; this file only renders the
   redacted state it receives and posts the player's actions.
   ============================================================ */

let CURRENT = null;   // latest full state from the server
let viewingScorecard = false;   // browsing the final scorecard after the match
let manualBracketView = false;  // tournament: player chose to leave a stale finished-match banner for the bracket

// Local, per-device UI state that must survive re-renders (polling
// re-renders on every version bump, incl. when the opponent submits).
const ui = {
    // Intent is remembered PER BATTER (by name), not per crease slot — ends
    // swap between every over (and can rotate mid-over), so a slot-keyed
    // value would silently reapply a stale setting to whoever is now
    // standing there instead of following the batter it was set for.
    batterIntents: {},
    bowlIntent: 50,
    // Role (Stage 4) replaces the intent slider. Batting role is per-batter (by
    // name, same reasoning as batterIntents); bowling role is a single value.
    batterRoles: {},
    bowlRole: 'contain',
    selectedBowler: null,
    openerPicks: [],
    expandedSquad: null,   // tournament auction: which "other squad" row is expanded, by team_id
    armGambit: false,      // one-shot gambit toggled on for the upcoming over submission
    impactPick: { in: null, out: null },   // Impact Player overlay: currently-selected in/out names
    roleHelpOpen: false,   // the little "what do the roles do?" popover
    farmStrike: false,     // "farm the strike" for the upcoming over (batting side)
};

function getBatterIntent(name) { return name in ui.batterIntents ? ui.batterIntents[name] : 50; }
function setBatterIntent(name, v) { ui.batterIntents[name] = v; }
function getBatterRole(name) { return name in ui.batterRoles ? ui.batterRoles[name] : 'rotate'; }
function setBatterRole(name, r) { ui.batterRoles[name] = r; }

// role definitions (label + which grid cell holds the 0-99 grade + a subtle
// accent color, reusing the site's existing palette -- no emoji, just a
// consistent color language: Attack = leather red, the "busy middle" role
// (Rotate/Contain) = gold, Defend = green, in both batting and bowling).
const BAT_ROLE_DEFS = [
    { key: 'attack', label: 'Attack', color: 'var(--leather)', cell: 'attack' },
    { key: 'rotate', label: 'Rotate', color: 'var(--gold)', cell: 'rotate' },
    { key: 'defend', label: 'Defend', color: 'var(--green-go)', cell: 'anchor' },
];
const BOWL_ROLE_DEFS = [
    { key: 'attack', label: 'Attack', color: 'var(--leather)', cell: 'attack' },
    { key: 'contain', label: 'Contain', color: 'var(--gold)', cell: 'contain' },
    { key: 'defend', label: 'Defend', color: 'var(--green-go)', cell: 'defend' },
];
function phaseKey(m) { return ({ powerplay: 'pp', middle: 'mid', death: 'death' })[m && m.phase_label] || 'mid'; }

// tracks the ball-by-ball reveal of the current over
const overAnim = { key: null, shown: 0, revealUntil: 0, timeouts: [] };

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);

function showScreen(id) {
    document.querySelectorAll('.screen').forEach(s => s.classList.toggle('active', s.id === id));
}

function toast(msg) {
    const t = document.createElement('div');
    t.className = 'toast';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2600);
}

function tierClass(ovr) {
    if (ovr >= 95) return 'tier-glow';     // glowing purple
    if (ovr >= 90) return 'tier-purple';   // purple
    if (ovr >= 80) return 'tier-gold';     // gold
    if (ovr >= 70) return 'tier-silver';   // silver
    if (ovr >= 60) return 'tier-bronze';   // bronze
    return 'tier-white';                   // 55-60 white
}

function intentWord(v) {
    if (v < 34) return 'Defensive';
    if (v > 66) return 'Aggressive';
    return 'Balanced';
}

// ---------- landing ----------
// Mirrors TEAM_COLOR_PALETTE in src/server.py -- keep in sync.
const TEAM_COLOR_PALETTE = ["#e6483c", "#3b82c4", "#e0b400", "#2f9e5c",
                            "#9d5ce0", "#e07b2e", "#22a6a6", "#e05c9e"];

function buildColorPicker(pickerId, hiddenId) {
    const root = $(pickerId);
    if (!root) return;
    root.innerHTML = TEAM_COLOR_PALETTE.map(c =>
        `<button type="button" class="color-swatch" data-color="${c}" style="background:${c}"></button>`).join('');
    root.querySelectorAll('.color-swatch').forEach(sw => {
        sw.addEventListener('click', () => {
            root.querySelectorAll('.color-swatch').forEach(s => s.classList.remove('picked'));
            sw.classList.add('picked');
            $(hiddenId).value = sw.dataset.color;
        });
    });
}
['create', 'join', 't-create', 't-join'].forEach(p => buildColorPicker(`${p}-color-picker`, `${p}-color`));

$('btn-create').addEventListener('click', async () => {
    try {
        const data = await Net.post('/api/create_game', {
            name: $('create-name').value.trim(), color: $('create-color').value || undefined,
        });
        Net.setToken(data.token);
        Net.startPolling();
    } catch (e) { $('landing-err').textContent = e.message; }
});

$('btn-join').addEventListener('click', async () => {
    const code = $('join-code').value.trim().toUpperCase();
    if (code.length !== 4) { $('landing-err').textContent = 'Enter the 4-character code.'; return; }
    try {
        const data = await Net.post('/api/join_game', {
            code, name: $('join-name').value.trim(), color: $('join-color').value || undefined,
        });
        Net.setToken(data.token);
        Net.startPolling();
    } catch (e) { $('landing-err').textContent = e.message; }
});

$('btn-quick').addEventListener('click', async () => {
    try { await Net.post('/api/quick_match', { token: Net.getToken() }); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
});

$('btn-auction').addEventListener('click', async () => {
    try { await Net.post('/api/start_auction', { token: Net.getToken() }); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
});

$('btn-create-tournament').addEventListener('click', async () => {
    try {
        const data = await Net.post('/api/create_tournament', {
            name: $('t-create-name').value.trim(),
            size: parseInt($('t-size').value, 10),
            color: $('t-create-color').value || undefined,
        });
        Net.setToken(data.token);
        Net.startPolling();
    } catch (e) { $('t-landing-err').textContent = e.message; }
});

$('btn-join-tournament').addEventListener('click', async () => {
    const code = $('t-join-code').value.trim().toUpperCase();
    if (code.length !== 4) { $('t-landing-err').textContent = 'Enter the 4-character code.'; return; }
    try {
        const data = await Net.post('/api/join_tournament', {
            code, name: $('t-join-name').value.trim(), color: $('t-join-color').value || undefined,
        });
        Net.setToken(data.token);
        Net.startPolling();
    } catch (e) {
        if (e.data && e.data.full && e.data.roster) {
            showRejoinPicker(code, e.data.roster);
        } else {
            $('t-landing-err').textContent = e.message;
        }
    }
});

function showRejoinPicker(code, roster) {
    $('rejoin-roster').innerHTML = roster.map(r =>
        `<div class="t-roster-slot joined clickable" data-rejoin="${r.team_id}">
            <div class="tname">${r.name}</div>
            <div class="tstatus">Tap to rejoin as this team</div>
        </div>`).join('');
    document.querySelectorAll('#rejoin-roster [data-rejoin]').forEach(el => {
        el.addEventListener('click', async () => {
            try {
                const data = await Net.post('/api/join_tournament', { code, rejoin_team_id: el.dataset.rejoin });
                Net.setToken(data.token);
                $('rejoin-overlay').classList.add('hidden');
                Net.startPolling();
            } catch (e) { $('t-landing-err').textContent = e.message; }
        });
    });
    $('rejoin-overlay').classList.remove('hidden');
}

$('btn-rejoin-cancel').addEventListener('click', () => $('rejoin-overlay').classList.add('hidden'));

$('btn-auction-rules').addEventListener('click', () => $('auction-rules-overlay').classList.remove('hidden'));
$('btn-close-auction-rules').addEventListener('click', () => $('auction-rules-overlay').classList.add('hidden'));
$('btn-close-fixture-scorecard').addEventListener('click', () => $('fixture-scorecard-overlay').classList.add('hidden'));
$('btn-skip-reveal').addEventListener('click', skipReveal);

$('btn-t-start').addEventListener('click', async () => {
    try { await Net.post('/api/start_auction', { token: Net.getToken() }); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
});

$('btn-ready-play').addEventListener('click', () => showScreen('landing'));

$('exit-btn').addEventListener('click', async () => {
    const finished = CURRENT && CURRENT.phase === 'finished';
    const isTourney = CURRENT && CURRENT.is_tournament;
    const warning = isTourney
        ? 'Leave the tournament? You will forfeit your current match and be eliminated — everyone else keeps playing.'
        : 'Leave the match? Your opponent will win.';
    if (!finished && !confirm(warning)) return;
    if (!finished) { try { await Net.post('/api/exit_game', { token: Net.getToken() }); } catch (e) { /* leaving anyway */ } }
    localStorage.removeItem('ca_token');
    location.reload();
});

$('cancel-tournament-btn').addEventListener('click', async () => {
    if (!confirm('Cancel the WHOLE tournament for everyone? This cannot be undone.')) return;
    try { await Net.post('/api/cancel_tournament', { token: Net.getToken() }); } catch (e) { /* ending anyway */ }
    localStorage.removeItem('ca_token');
    location.reload();
});

$('btn-new-game').addEventListener('click', () => { localStorage.removeItem('ca_token'); location.reload(); });

$('btn-back-to-tournament').addEventListener('click', () => {
    manualBracketView = true;
    $('result-banner').classList.add('hidden');
    showScreen('t-bracket');
    if (CURRENT) renderBracket(CURRENT);
});

$('back-to-tournament-fixed-btn').addEventListener('click', () => {
    viewingScorecard = false;
    manualBracketView = true;
    $('result-banner').classList.add('hidden');
    showScreen('t-bracket');
    if (CURRENT) renderBracket(CURRENT);
});

$('btn-view-scorecard').addEventListener('click', () => {
    viewingScorecard = true;
    $('result-banner').classList.add('hidden');
    $('exit-btn').classList.remove('hidden');
    document.querySelectorAll('#game .tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'scorecard'));
    document.querySelectorAll('#game .tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-scorecard'));
});

// ---------- tabs ----------
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        if (tab.dataset.atab) {
            const name = tab.dataset.atab;
            document.querySelectorAll('.auc-tab').forEach(t => t.classList.toggle('active', t === tab));
            document.querySelectorAll('#auction .tab-panel').forEach(p => p.classList.toggle('active', p.id === `atab-${name}`));
        } else if (tab.dataset.tab) {
            const name = tab.dataset.tab;
            document.querySelectorAll('#game .tab').forEach(t => t.classList.toggle('active', t === tab));
            document.querySelectorAll('#game .tab-panel').forEach(p => p.classList.toggle('active', p.id === `tab-${name}`));
        }
    });
});

// ---------- auction floor: delegated listeners (registered once; the
// tables/menu are rebuilt on every poll, so per-element listeners would be
// lost -- these live on document/static containers instead) ----------
document.addEventListener('mouseover', (e) => {
    const t = e.target.closest && e.target.closest('.round-table');
    if (t) showTableInteractFor(t);
});
document.addEventListener('mouseout', (e) => {
    const t = e.target.closest && e.target.closest('.round-table');
    const toEl = e.relatedTarget && e.relatedTarget.closest
        && e.relatedTarget.closest('.round-table, #auc-table-interact');
    if (t && !toEl) scheduleHideInteract();
});
document.addEventListener('click', (e) => {
    const t = e.target.closest && e.target.closest('.round-table');
    if (t) openSquadPopup(t.dataset.team);
});
if ($('auc-table-interact')) {
    $('auc-table-interact').addEventListener('mouseenter', () => clearTimeout(interactHideTimer));
    $('auc-table-interact').addEventListener('mouseleave', scheduleHideInteract);
    $('auc-table-interact').addEventListener('click', (e) => {
        const pokeBtn = e.target.closest('[data-poke]');
        if (pokeBtn && !pokeBtn.disabled) { aucAction('/api/auction_poke', { target: pokeBtn.dataset.poke }); return; }
        const toggleBtn = e.target.closest('[data-banter-toggle]');
        if (toggleBtn) {
            const pop = document.getElementById('banter-pop');
            if (pop.classList.contains('open')) { pop.classList.remove('open'); return; }
            pop.innerHTML = BANTER_LINES.map((l, i) => `<button data-banter-line="${i}">${l}</button>`).join('');
            pop.classList.add('open');
            return;
        }
        const lineBtn = e.target.closest('[data-banter-line]');
        if (lineBtn) {
            const target = $('auc-table-interact').dataset.team;
            aucAction('/api/auction_banter', { target, line_index: parseInt(lineBtn.dataset.banterLine, 10) });
            document.getElementById('banter-pop').classList.remove('open');
        }
    });
}
if ($('squad-popup-overlay')) {
    $('squad-popup-overlay').addEventListener('click', (e) => {
        if (e.target.id === 'squad-popup-overlay') closeSquadPopup();
    });
}

// ---------- master render ----------
window.addEventListener('gamestate', (e) => {
    CURRENT = e.detail;
    render(CURRENT);
});

function updateRejoinLabel(state) {
    const el = $('rejoin-code-label');
    // cleared on every render; re-applied below only while "Back to Tournament"
    // is actually up, so it can't get stuck lifted after that button goes away
    el.classList.remove('stacked');
    if (state && state.code) {
        $('rejoin-code-value').textContent = state.code;
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

let lobbyPrevPhase = null;
let lobbyDoorsPlayed = false;

function render(state) {
    if (!state || state.status === 'no_game' || !state.you || !state.you.joined) {
        $('exit-btn').classList.add('hidden');
        $('cancel-tournament-btn').classList.add('hidden');
        $('back-to-tournament-fixed-btn').classList.add('hidden');
        updateRejoinLabel(null);
        showScreen('landing');
        lobbyPrevPhase = null;
        lobbyDoorsPlayed = false;
        return;
    }
    // Elevator doors: play once, exactly on the lobby -> auction phase switch --
    // the join lobby (both plain and tournament) IS the elevator now; by the
    // time phase is 'auction' you're already meant to be "in the room", so
    // the auction screen itself never re-shows the lobby (see renderAuction).
    if (lobbyPrevPhase === 'lobby' && state.phase !== 'lobby' && !lobbyDoorsPlayed) {
        playLobbyDoorsTransition();
        lobbyDoorsPlayed = true;
    }
    lobbyPrevPhase = state.phase;

    updateRejoinLabel(state);
    // opponent (or you) left -> show result, offer back to main
    if (state.abandoned) {
        $('exit-btn').classList.add('hidden');
        $('cancel-tournament-btn').classList.add('hidden');
        $('back-to-tournament-fixed-btn').classList.add('hidden');
        $('result-text').textContent = state.you_won
            ? 'You win! Your opponent left the match.'
            : (state.ended_result || 'Match abandoned.');
        $('result-motm').classList.add('hidden');
        $('result-banner').classList.remove('hidden');
        return;
    }
    // squad never reached the minimum by the grace deadline -> kicked out
    if (state.eliminated && !manualBracketView) {
        $('exit-btn').classList.add('hidden');
        $('cancel-tournament-btn').classList.toggle('hidden', !state.is_tournament);
        $('back-to-tournament-fixed-btn').classList.add('hidden');
        $('result-text').textContent = state.ended_result || "Your squad never reached the minimum — you're out.";
        $('result-motm').classList.add('hidden');
        $('result-banner').classList.remove('hidden');
        $('btn-new-game').classList.toggle('hidden', state.is_tournament);
        $('btn-back-to-tournament').classList.toggle('hidden', !state.is_tournament);
        return;
    }
    // exit/leave button: shown during play and while browsing the final
    // scorecard; hidden only while the result banner itself is up.
    const finished = state.phase === 'finished';
    $('exit-btn').classList.toggle('hidden', finished && !viewingScorecard);

    // cancel-the-whole-tournament button: visible anywhere in a tournament
    // that hasn't concluded yet (distinct from exit-btn, which only forfeits
    // YOUR current fixture and lets everyone else keep playing)
    const tournamentOver = state.tournament && state.tournament.stage === 'champion';
    $('cancel-tournament-btn').classList.toggle('hidden', !state.is_tournament || tournamentOver);

    // escape hatch: browsing the scorecard while some OTHER fixture plays on
    // elsewhere used to leave you stranded there with no way back (the
    // automatic "waiting_for_fixture" redirect only kicks in once the other
    // fixture actually starts, which can lag well behind your own finishing)
    const backBtnUp = state.is_tournament && viewingScorecard;
    $('back-to-tournament-fixed-btn').classList.toggle('hidden', !backBtnUp);
    // both are pinned to the bottom-right corner -- lift the code pill clear
    $('rejoin-code-label').classList.toggle('stacked', !!backBtnUp);

    // manualBracketView only makes sense while our own last fixture is still
    // the "finished" one sitting behind it (including if a next-fixture
    // ready-prompt has since queued up — that now lives IN the bracket
    // screen, see renderBracket) — clear it once a genuinely new phase
    // starts for us so future banners behave normally again.
    if (!(finished || state.eliminated)) {
        manualBracketView = false;
    }

    if (state.phase === 'lobby') {
        if (state.is_tournament) { showScreen('t-lobby'); renderTournamentLobby(state); }
        else { showScreen('lobby'); renderLobby(state); }
        return;
    }
    if (state.phase === 'auction') { showScreen('auction'); renderAuction(state); return; }
    if (state.phase === 'grounds') { showScreen('grounds-screen'); renderGrounds(state); return; }
    if (state.phase === 'xi') { showScreen('xi-screen'); renderXI(state); return; }
    // tournament: fresh per-fixture XI reselect (not the one-time phase=='xi' above)
    if (state.fixture_xi) { showScreen('xi-screen'); renderXI(state); return; }

    // tournament: champion crowned takes priority over everything else
    if (state.is_tournament && state.tournament && state.tournament.stage === 'champion' && !viewingScorecard) {
        showScreen('t-bracket');
        renderChampion(state);
        return;
    }
    // tournament: this team isn't in the fixture currently live -> show the bracket
    if (state.is_tournament && state.waiting_for_fixture) {
        showScreen('t-bracket');
        renderBracket(state);
        return;
    }
    // tournament: our own fixture just ended -> ALWAYS land back on the
    // bracket automatically (no manual "Back to Tournament" click required,
    // and no separate scorecard screen to get stranded on — the just-played
    // match's result/MOTM shows inline, and its full scorecard is one tap
    // away via the same fixture-scorecard popup as any other match).
    if (finished && state.is_tournament) {
        showScreen('t-bracket');
        renderBracket(state);
        return;
    }

    showScreen('game');
    renderGame(state);

    // hold off popping the (opaque, full-screen) result banner until the
    // final over's ball-by-ball reveal has actually finished playing out —
    // otherwise it covers the commentary the instant the match ends and it
    // looks like no commentary was ever shown at all.
    const revealDone = Date.now() >= overAnim.revealUntil;
    if (finished && !viewingScorecard && revealDone) {
        $('result-text').textContent = (state.match && state.match.result) || 'Match complete';
        const motm = state.match && state.match.motm;
        $('result-motm').textContent = motm ? `Player of the Match: ${motm}` : '';
        $('result-motm').classList.toggle('hidden', !motm);
        $('result-banner').classList.remove('hidden');
        const isChampionDone = state.is_tournament && state.tournament && state.tournament.stage === 'champion';
        // "New Game" ends the WHOLE session — only offer it for a plain 1v1
        // match or once the tournament is truly over. Mid-tournament, offer
        // "Back to Tournament" instead so nobody accidentally wipes their
        // token thinking one fixture ending means the whole thing is done.
        $('btn-new-game').classList.toggle('hidden', state.is_tournament && !isChampionDone);
        $('btn-back-to-tournament').classList.toggle('hidden', !state.is_tournament || isChampionDone);
    } else {
        $('result-banner').classList.add('hidden');
    }
}

// ---------- lobby: the elevator waiting room, shown from the moment you
// join with a code until everyone presses ready and the doors open into the
// auction. Shared between the plain 1v1 lobby and the tournament lobby --
// only where the guest list/ready-state comes from differs. ----------
function elevatorLobbyHtml(code, guests, statusText) {
    const guestsHtml = guests.map(g => `
        <div class="lob-guest ${g.state}">
            <div class="lob-avatar">${g.state === 'empty' ? '?' : initials(g.name)}</div>
            <div class="lob-guest-name">${g.name}</div>
            <div class="lob-dot"></div>
        </div>`).join('');
    return `
    <div class="lobby-mockup">
        <div class="lob-code-badge">Share this code: <b>${code}</b></div>
        <div class="lob-walls">
            <div class="lob-indicator"><span class="lob-ind-label">Now Boarding</span><span class="lob-ind-value">Auction Floor</span></div>
            <div class="lob-doors">
                <div class="lob-beyond"><div class="lob-beyond-glow"></div><div class="lob-beyond-tables"><div></div><div></div><div></div></div></div>
                <div class="lob-door lob-door-l"><div class="lob-door-emblem"></div></div>
                <div class="lob-door lob-door-r"><div class="lob-door-emblem"></div></div>
            </div>
        </div>
        <div class="lob-guests">${guestsHtml}</div>
        <div class="lob-status">${statusText}</div>
        <div class="lob-floor"></div>
    </div>`;
}

function renderLobby(state) {
    const lob = state.lobby || {};
    const guests = [
        { name: state.you.name || 'You', state: lob.i_voted ? 'ready' : 'joined' },
        state.opponent.joined
            ? { name: state.opponent.name, state: lob.opponent_voted ? 'ready' : 'joined' }
            : { name: 'Waiting to join…', state: 'empty' },
    ];
    const both = state.you.joined && state.opponent.joined;
    const statusText = both
        ? `${guests.filter(g => g.state === 'ready').length} of 2 ready`
        : 'Waiting for opponent to join…';
    $('lobby-elevator').innerHTML = elevatorLobbyHtml(state.code, guests, statusText);

    $('lobby-actions').classList.toggle('hidden', !both);
    const btn = $('btn-auction');
    if (lob.i_voted) { btn.textContent = 'Waiting for opponent…'; btn.disabled = true; }
    else if (lob.opponent_voted) { btn.textContent = "Ready for Auction (opponent's in)"; btn.disabled = false; }
    else { btn.textContent = 'Ready for Auction'; btn.disabled = false; }
}

// ---------- tournament lobby ----------
function renderTournamentLobby(state) {
    const tl = state.tournament_lobby;
    const guests = tl.roster.map(r => ({
        name: r.joined ? r.name : 'Waiting to join…',
        state: !r.joined ? 'empty' : (tl.start_votes[r.team_id] ? 'ready' : 'joined'),
    }));
    const statusText = tl.all_joined
        ? `${guests.filter(g => g.state === 'ready').length} of ${tl.size} ready`
        : `${tl.joined_count}/${tl.size} teams joined`;
    $('t-lobby-elevator').innerHTML = elevatorLobbyHtml(state.code, guests, statusText);

    const btn = $('btn-t-start');
    btn.classList.toggle('hidden', !tl.all_joined);
    if (tl.all_joined) {
        if (tl.i_voted) { btn.textContent = 'Waiting for everyone…'; btn.disabled = true; }
        else { btn.textContent = 'Ready for Auction'; btn.disabled = false; }
    }
}

// ---------- tournament bracket / waiting / champion ----------
function standingsTable(standings, highlightTop) {
    if (!standings || !standings.length) return '';
    const rows = standings.map((s, i) => `
        <tr class="${i < highlightTop ? 'qualifying' : ''}">
            <td>${i + 1}. ${s.name}</td>
            <td>${s.played}</td><td>${s.won}</td><td>${s.lost}</td>
            <td>${s.points}</td><td>${s.nrr >= 0 ? '+' : ''}${s.nrr.toFixed(2)}</td>
        </tr>`).join('');
    return `<div class="t-standings"><table>
        <thead><tr><th>Team</th><th>P</th><th>W</th><th>L</th><th>Pts</th><th>NRR</th></tr></thead>
        <tbody>${rows}</tbody>
    </table></div>`;
}

function fixturesList(fixtures) {
    if (!fixtures || !fixtures.length) return '';
    const rows = fixtures.map(f => `
        <div class="t-fixture-row ${f.played ? '' : 'pending'}${f.has_scorecard ? ' clickable' : ''}"
             ${f.has_scorecard ? `data-fixture-idx="${f.idx}"` : ''}>
            <span><span class="kind">${f.kind.replace('_', ' ')}</span> ${f.a_name} vs ${f.b_name}${f.host_name ? ` <span class="host-tag" title="Home ground">Host: ${f.host_name}</span>` : ''}</span>
            <span>${f.played ? (f.result_text || '') : 'upcoming'}${f.motm_name ? ` &middot; MOTM: ${f.motm_name}` : ''}
                ${f.has_scorecard ? ' &middot; Scorecard' : ''}</span>
        </div>`).join('');
    return `<div class="t-fixtures">${rows}</div>`;
}

function wireFixtureScorecards() {
    document.querySelectorAll('[data-fixture-idx]').forEach(el =>
        el.addEventListener('click', () => openFixtureScorecard(el.dataset.fixtureIdx)));
}

async function openFixtureScorecard(idx) {
    try {
        const r = await Net.get(`/api/fixture_scorecard?token=${Net.getToken()}&fixture_idx=${idx}`);
        if (r.status !== 'success') { toast(r.message || 'No scorecard available.'); return; }
        $('fixture-scorecard-title').textContent = `${r.a_name} vs ${r.b_name} — ${r.kind.replace('_', ' ')}`;
        $('fixture-scorecard-sub').textContent = r.result_text || '';
        renderScorecardInto('fixture-scorecard-body', r.innings);
        $('fixture-scorecard-overlay').classList.remove('hidden');
    } catch (e) { toast(e.message); }
}

function renderBracket(state) {
    const t = state.tournament;
    let html = `<div class="tagline" style="text-align:center;">Tournament Bracket</div>`;
    // just came straight back from our own finished fixture (no more manual
    // "Back to Tournament" click needed — see render()) -> show the result
    // and MOTM inline, with the full scorecard one tap away
    if (state.phase === 'finished' && state.match && state.match.result) {
        html += lastResultBlock(state.match, t.current_fixture_idx);
    }
    // your match is next -> ready-up prompt lives HERE now (not a floating
    // overlay stacked over the previous match's result banner), so you only
    // see it once you've actually come back to the tournament screen
    if (state.next_fixture) {
        html += nextFixtureBlock(state.next_fixture);
    }
    if (t.current_fixture) {
        html += `<div class="t-current-fixture">
            <div class="tagline">Now Playing</div>
            <div>${t.current_fixture.a_name} <span class="vs">VS</span> ${t.current_fixture.b_name}</div>
            <div class="tagline" style="margin-top:0.3rem;">${t.current_fixture.kind.replace('_', ' ')}${t.current_fixture.host_name ? ` &middot; Host: ${t.current_fixture.host_name}` : ''}</div>
            ${spectateBlock(state.spectate)}
        </div>`;
    }
    html += awardsLeaderboard(t.awards);
    html += standingsTable(t.standings, 3);
    html += myRosterEnergy(state.my_roster);
    html += fixturesList(t.fixtures);
    $('t-bracket-wrap').innerHTML = html;
    wireBracketActions();
    wireFixtureScorecards();
    wireAwardsExpand();
    wireRosterEnergyExpand();
    // spectateBlock only left a placeholder div for the scorecard table (it
    // returns an HTML string, it can't run the table-building DOM code itself)
    if (state.spectate && state.spectate.scorecard) {
        renderScorecardInto('spectate-scorecard', state.spectate.scorecard);
    }
}

// ---------- squad energy tab (tournament home screen) ----------
let rosterEnergyExpanded = false;
function myRosterEnergy(roster) {
    if (!roster) return '';
    const rows = roster.players
        .slice()
        .sort((a, b) => a.energy - b.energy || a.name.localeCompare(b.name))
        .map(p => {
            const level = p.energy >= 99 ? 'fresh' : p.energy >= 97 ? 'tired' : 'worn';
            return `<div class="roster-energy-row ${level}">
                <span class="re-name">${p.name} ${p.is_foreigner ? '<span class="os">OS</span>' : ''}${p.is_keeper ? ' <span class="os">WK</span>' : ''}</span>
                <span class="re-bar"><span class="re-bar-fill" style="width:${p.energy}%"></span></span>
                <span class="re-value">${p.energy}%</span>
            </div>`;
        }).join('');
    return `<div class="t-roster-energy">
        <div class="tagline t-roster-energy-toggle" id="roster-energy-toggle" style="text-align:center; cursor:pointer;">
            ${roster.team_name} — Squad Energy ${rosterEnergyExpanded ? '[-]' : '[+]'}
        </div>
        <div class="t-roster-energy-body${rosterEnergyExpanded ? '' : ' hidden'}">
            <div class="tagline" style="text-align:center; opacity:0.7; margin-bottom:0.4rem;">
                Every player starts each tournament at 100% energy. Playing a match costs 0.5% (floor 95%);
                resting a match fully restores it.
            </div>
            ${rows}
        </div>
    </div>`;
}

function wireRosterEnergyExpand() {
    const el = $('roster-energy-toggle');
    if (el) el.addEventListener('click', () => {
        rosterEnergyExpanded = !rosterEnergyExpanded;
        if (CURRENT) render(CURRENT);
    });
}

const AWARDS_COLLAPSED_COUNT = 3;
const awardsExpanded = { orange_cap: false, purple_cap: false, mvp: false };

function awardsLeaderboardList(key, entries, unit) {
    const expanded = awardsExpanded[key];
    const shown = expanded ? entries : entries.slice(0, AWARDS_COLLAPSED_COUNT);
    const rows = shown.map((e, i) => `
        <div class="t-award-row">
            <span class="t-award-rank">${i + 1}</span>
            <span class="t-award-pname">${e.name}</span>
            <span class="t-award-pteam">${e.team_name}</span>
            <span class="t-award-pvalue">${e.value}${unit ? ' ' + unit : ''}</span>
        </div>`).join('');
    const hidden = entries.length - AWARDS_COLLAPSED_COUNT;
    const hint = hidden > 0
        ? `<div class="t-award-expand" data-award-key="${key}">${expanded ? 'show less' : `show ${hidden} more`}</div>`
        : '';
    return rows + hint;
}

function awardsLeaderboard(awards) {
    if (!awards) return '';
    return `<div class="t-awards">
        <div class="tagline" style="text-align:center;">Running Leaderboard</div>
        <div class="t-awards-row">
            <div class="t-award-card"><div class="t-award-cap">Orange Cap</div>
                ${awardsLeaderboardList('orange_cap', awards.orange_cap, 'runs')}</div>
            <div class="t-award-card"><div class="t-award-cap">Purple Cap</div>
                ${awardsLeaderboardList('purple_cap', awards.purple_cap, 'wkts')}</div>
            <div class="t-award-card"><div class="t-award-cap">MVP</div>
                ${awardsLeaderboardList('mvp', awards.mvp, '')}</div>
        </div>
    </div>`;
}

function wireAwardsExpand() {
    document.querySelectorAll('[data-award-key]').forEach(el =>
        el.addEventListener('click', () => {
            const key = el.dataset.awardKey;
            awardsExpanded[key] = !awardsExpanded[key];
            if (CURRENT) render(CURRENT);
        }));
}

function lastResultBlock(m, fixtureIdx) {
    return `<div class="t-last-result">
        <div class="tagline">Your Match</div>
        <div class="t-last-result-text">${m.result}</div>
        ${m.motm ? `<div class="t-last-result-motm">Player of the Match: ${m.motm}</div>` : ''}
        ${fixtureIdx !== null && fixtureIdx !== undefined
            ? `<button class="btn-ghost" id="btn-last-result-scorecard" data-fixture-idx="${fixtureIdx}">View Scorecard</button>`
            : ''}
    </div>`;
}

function nextFixtureBlock(nf) {
    if (nf.i_ready) {
        return `<div class="t-next-fixture">
            <div class="tagline">Your Match Is Next — Ready</div>
            <div>${nf.a_name} <span class="vs">VS</span> ${nf.b_name}</div>
            <div class="tagline" style="margin-top:0.3rem; opacity:0.75;">
                Waiting for ${nf.opponent_ready ? 'the match to start…' : 'the other team to ready up…'}</div>
        </div>`;
    }
    return `<div class="t-next-fixture">
        <div class="tagline">Your Match Is Next</div>
        <div>${nf.a_name} <span class="vs">VS</span> ${nf.b_name}</div>
        <div class="tagline" style="margin-top:0.3rem;">${nf.kind.replace('_', ' ')}</div>
        <button class="btn-go btn-lg" id="btn-fixture-ready" style="margin-top:0.8rem;">I'm Ready</button>
    </div>`;
}

// Non-playing tournament teams get the full match, not just a score line:
// live player cards for whoever's at the crease/bowling, extras/target, the
// ball-by-ball over, and the complete scorecard below (spectateBlock only
// returns the HTML string -- the scorecard table itself is a DOM-building
// function, wired in separately right after this gets inserted; see
// renderBracket).
function spectateBlock(sp) {
    if (!sp) return '';
    const overLines = (sp.this_over || []).map(commLine).join('') || '<div class="comm-empty">Over about to start…</div>';
    const cards = [
        sp.striker ? pcard(sp.striker, { figure: `${sp.striker.runs}(${sp.striker.balls})`,
            jerseyColor: sp.batting_team_color, jerseyStyle: sp.batting_team_jersey }) : '',
        sp.non_striker ? pcard(sp.non_striker, { figure: `${sp.non_striker.runs}(${sp.non_striker.balls})`,
            jerseyColor: sp.batting_team_color, jerseyStyle: sp.batting_team_jersey }) : '',
        sp.bowler ? pcard(sp.bowler, { figure: `${sp.bowler.wickets}-${sp.bowler.runs}(${sp.bowler.overs})`,
            jerseyColor: sp.bowling_team_color, jerseyStyle: sp.bowling_team_jersey }) : '',
    ].filter(Boolean).join('');
    return `<div class="t-spectate">
        <div class="t-spectate-score">${sp.batting_team_name} ${sp.score}/${sp.wickets}
            <span class="t-spectate-overs">(${sp.overs} ov)</span></div>
        ${sp.ground ? `<div class="t-spectate-sub">${sp.ground.name} &middot; <span class="pitch-chip ${sp.ground.pitch}">${sp.ground.pitch}</span></div>` : ''}
        <div class="t-spectate-sub">Extras ${sp.extras}${sp.target ? ` &middot; Target ${sp.target}` : ''}</div>
        <div class="t-spectate-cards">${cards}</div>
        <div class="comm-list t-spectate-comm">${overLines}</div>
        <div class="t-spectate-sub" style="margin-top:0.6rem;">Full Scorecard</div>
        <div id="spectate-scorecard"></div>
    </div>`;
}

function wireBracketActions() {
    const b = $('btn-fixture-ready');
    if (b) b.addEventListener('click', async () => {
        try { await Net.post('/api/tournament_ready', { token: Net.getToken() }); Net.forceRefresh(); }
        catch (e) { toast(e.message); }
    });
}

// one-time post-tournament awards reveal, played once per page load (a new
// tournament always means a fresh page load via location.reload())
let presentationShown = false;
let presentationInProgress = false;

function renderChampion(state) {
    if (presentationInProgress) return;   // let the reveal sequence own the DOM until it's done
    if (!presentationShown) {
        presentationShown = true;
        presentationInProgress = true;
        playVictoryFanfare();
        showPresentationSequence(state);
        return;
    }
    renderChampionScreen(state);
}

function showPresentationSequence(state) {
    const t = state.tournament;
    const steps = [];
    const top3 = (list, unit) => (list || []).slice(0, 3).map((e, i) =>
        `<div class="t-presentation-rank">${i + 1}. ${e.name} <span>(${e.team_name} · ${e.value}${unit ? ' ' + unit : ''})</span></div>`
    ).join('');
    if (t.awards) {
        steps.push({ label: 'Orange Cap — Most Runs', listHtml: top3(t.awards.orange_cap, 'runs') });
        steps.push({ label: 'Purple Cap — Most Wickets', listHtml: top3(t.awards.purple_cap, 'wkts') });
        steps.push({ label: 'Tournament MVP', listHtml: top3(t.awards.mvp, '') });
    }
    steps.push({ label: 'Tournament Champions', name: t.champion_name, sub: 'Congratulations!', big: true });

    let i = 0;
    const showStep = () => {
        if (i >= steps.length) {
            presentationInProgress = false;
            renderChampionScreen(CURRENT || state);
            return;
        }
        const s = steps[i];
        $('t-bracket-wrap').innerHTML = `
            <div class="t-presentation${s.big ? ' big' : ''}">
                <div class="t-presentation-title">${s.label}</div>
                ${s.big ? `<h1>${s.name}</h1><div class="tagline">${s.sub}</div>` : `<div class="t-presentation-list">${s.listHtml}</div>`}
            </div>`;
        i++;
        setTimeout(showStep, s.big ? 2800 : 2200);
    };
    showStep();
}

function renderChampionScreen(state) {
    const t = state.tournament;
    // only the two finalists actually received match/scorecard data from the
    // server (spectators are redacted to the bracket summary only)
    const canViewScorecard = !!state.match;
    $('t-bracket-wrap').innerHTML = `
        <div class="t-champion-banner">
            <h1>${t.champion_name}</h1>
            <div class="tagline">TOURNAMENT CHAMPIONS</div>
        </div>
        ${awardsLeaderboard(t.awards)}
        ${standingsTable(t.standings, 3)}
        ${fixturesList(t.fixtures)}
        <div style="display:flex; gap:0.8rem; margin-top:1rem;">
            ${canViewScorecard ? '<button class="btn-ghost btn-lg" id="btn-t-view-final">View Final Scorecard</button>' : ''}
            <button class="btn-gold btn-lg" onclick="localStorage.removeItem('ca_token'); location.reload();">New Tournament</button>
        </div>`;
    const vb = document.getElementById('btn-t-view-final');
    if (vb) vb.addEventListener('click', () => {
        viewingScorecard = true;
        showScreen('game');
        renderGame(CURRENT);
        document.querySelectorAll('#game .tab').forEach(tb => tb.classList.toggle('active', tb.dataset.tab === 'scorecard'));
        document.querySelectorAll('#game .tab-panel').forEach(p => p.classList.toggle('active', p.id === 'tab-scorecard'));
    });
    wireFixtureScorecards();
    wireAwardsExpand();
}

// ---------- game shell ----------
function renderGame(state) {
    const m = state.match;

    // TOSS — happens before any innings state exists
    if (m.stage === 'toss') {
        $('role-pill').textContent = 'Toss';
        $('role-pill').className = 'role-pill';
        renderToss(m);
        return;
    }
    hideOverlay('toss-overlay');

    $('role-pill').textContent = m.i_am_batting ? 'Batting' : 'Bowling';
    $('role-pill').className = 'role-pill ' + (m.i_am_batting ? 'batting' : 'bowling');

    // reset stale bowler selection if it's no longer valid on the bench
    if (!m.i_am_batting) {
        const legal = m.my_bench.filter(b => !b.disabled).map(b => b.name);
        if (ui.selectedBowler && !legal.includes(ui.selectedBowler)) ui.selectedBowler = null;
    }
    // adopt server-locked roles once submitted (so both sides agree post-reveal)
    if (m.pending.i_submitted) {
        const mine = m.pending.mine;
        if (m.i_am_batting) {
            if (m.striker) setBatterRole(m.striker.name, mine.striker_role);
            if (m.non_striker) setBatterRole(m.non_striker.name, mine.non_striker_role);
            ui.farmStrike = !!m.farm_strike;   // server is the source of truth once locked
        } else { ui.bowlRole = mine.bowl_role; ui.selectedBowler = mine.bowler_name; }
    }

    renderScoreboard(m);
    renderGround(m);
    renderReadyBar(m);
    // keep the Impact Player overlay's picks in sync if it's open when a poll
    // lands (e.g. the pool/out-options shift, or the window closes entirely)
    if (!$('impact-overlay').classList.contains('hidden')) {
        if (m.impact && m.impact.can_use) renderImpactOverlay(m);
        else closeImpactOverlay();
    }
    renderBench(m);
    renderThisOver(m);
    renderOppList(m);
    renderLive(state.live);
    renderScorecard(state.scorecard);

    // OPENERS — batting side picks its opening pair
    if (m.stage === 'openers') renderOpeners(m); else hideOverlay('openers-overlay');
    // (result banner is managed in render() so the final scorecard can be browsed)
}

function hideOverlay(id) { $(id).classList.add('hidden'); }

// ---------- toss ----------
function renderToss(m) {
    const card = $('toss-card');
    const groundLine = m.ground
        ? `<div class="sub toss-ground">${m.ground.name}, ${m.ground.city}<br>
             <span class="pitch-chip ${m.ground.pitch}">${m.ground.pitch} pitch</span> — ${m.ground.pitch_desc}</div>`
        : '';
    if (m.toss.i_won) {
        card.innerHTML = `
            <div class="toss-coin">TOSS</div>
            <h2>You won the toss!</h2>
            ${groundLine}
            <div class="sub">${m.toss.winner_name}, what will you do?</div>
            <div class="toss-choices">
                <button class="btn-go btn-lg" data-toss="bat">Bat First</button>
                <button class="btn-red btn-lg" data-toss="bowl">Bowl First</button>
            </div>`;
        card.querySelectorAll('[data-toss]').forEach(b =>
            b.addEventListener('click', () => tossChoice(b.dataset.toss)));
    } else {
        card.innerHTML = `
            <div class="toss-coin">TOSS</div>
            <h2>Coin is in the air…</h2>
            ${groundLine}
            <div class="sub wait-note">${m.toss.winner_name} won the toss and is deciding to bat or bowl.</div>`;
    }
    $('toss-overlay').classList.remove('hidden');
}

async function tossChoice(choice) {
    try { await Net.post('/api/toss_choice', { token: Net.getToken(), choice }); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
}

// ---------- openers ----------
function renderOpeners(m) {
    const card = $('openers-card');
    if (!m.i_am_batting) {
        card.innerHTML = `<h2>Openers</h2>
            <div class="sub wait-note">${m.batting_team_name} is choosing their opening pair…</div>`;
        $('openers-overlay').classList.remove('hidden');
        return;
    }
    const avail = m.my_bench.filter(b => b.status === 'available');
    const grid = avail.map(b => {
        const picked = ui.openerPicks.indexOf(b.name);
        const badge = picked === 0 ? '<div class="pick-badge">STRIKER</div>'
            : picked === 1 ? '<div class="pick-badge">NON-STR</div>' : '';
        return pcard(b, {
            ovr: b.batting_ovr, selectable: true, selected: picked >= 0,
            attrs: `data-opener="${b.name}"`, extraHtml: badge,
            jerseyColor: m.batting_team_color, jerseyStyle: m.batting_team_jersey,
        });
    }).join('');
    const ready = ui.openerPicks.length === 2;
    card.innerHTML = `
        <h2>Pick Your Openers</h2>
        <div class="sub">Tap two batsmen — first pick takes strike.</div>
        <div class="openers-grid">${grid}</div>
        <button class="btn-go btn-lg" id="confirm-openers" ${ready ? '' : 'disabled'}>
            ${ready ? 'Send Them Out' : `Pick ${2 - ui.openerPicks.length} more`}</button>`;
    card.querySelectorAll('[data-opener]').forEach(el =>
        el.addEventListener('click', () => toggleOpener(el.dataset.opener)));
    const cbtn = $('confirm-openers');
    if (cbtn) cbtn.addEventListener('click', confirmOpeners);
    $('openers-overlay').classList.remove('hidden');
}

function toggleOpener(name) {
    const i = ui.openerPicks.indexOf(name);
    if (i >= 0) ui.openerPicks.splice(i, 1);
    else if (ui.openerPicks.length < 2) ui.openerPicks.push(name);
    renderOpeners(CURRENT.match);
}

async function confirmOpeners() {
    if (ui.openerPicks.length !== 2) return;
    try {
        await Net.post('/api/set_openers', {
            token: Net.getToken(),
            striker: ui.openerPicks[0], non_striker: ui.openerPicks[1],
        });
        ui.openerPicks = [];
        Net.forceRefresh();
    } catch (e) { toast(e.message); }
}

// ---------- scoreboard ----------
function renderScoreboard(m) {
    let target = '';
    if (m.target) {
        const need = m.target - m.runs;
        const ballsLeft = m.max_overs * 6 - m.balls;
        target = `<div class="sb-target"><div class="sb-meta">TARGET</div>
            <div class="big">${m.target}</div>
            <div class="sb-meta">${need > 0 ? `Need ${need} off ${ballsLeft}` : 'Chased!'}</div></div>`;
    } else {
        target = `<div class="sb-target"><div class="sb-meta">1st Innings</div></div>`;
    }
    const phaseChip = m.phase_label && m.phase_label !== 'middle'
        ? `<span class="phase-chip ${m.phase_label}">${m.phase_label === 'powerplay' ? 'POWERPLAY' : 'DEATH OVERS'}</span>` : '';
    $('scoreboard').innerHTML = `
        <div>
            <div class="sb-team" style="${m.batting_team_color ? `color:${m.batting_team_color}` : ''}">${m.batting_team_name}</div>
            <div class="sb-score">${m.runs}/${m.wickets}</div>
            <div class="sb-meta">Overs ${m.overs} / ${m.max_overs} &middot; Extras ${m.extras}</div>
            ${m.ground ? `<div class="sb-meta sb-ground">${m.ground.name} &middot; <span class="pitch-chip ${m.ground.pitch}">${m.ground.pitch} pitch</span></div>` : ''}
        </div>
        <div class="sb-meta">bowling: <span style="${m.bowling_team_color ? `color:${m.bowling_team_color}` : ''}">${m.bowling_team_name}</span> ${phaseChip}</div>
        ${target}
        ${roleHelpHtml(m.i_am_batting)}`;
    wireRoleHelp();
}

// Jersey styling (team color + home/away) for a pcard -- the collar band
// across the top (solid for home, hollow for away) plus a faint team-color
// wash over the whole card body. Purely cosmetic, no gameplay meaning.
function jerseyAttrs(color, style) {
    if (!color) return { attr: '', style: '' };
    const s = style === 'away' ? 'away' : 'home';
    const washTop = s === 'home' ? color + '40' : color + '22';
    const washBot = s === 'home' ? color + '1a' : color + '0d';
    return {
        attr: ` data-jersey data-jersey-style="${s}"`,
        style: `--jc:${color}; --jc-tint:${color}26; --jc-wash-top:${washTop}; --jc-wash-bot:${washBot};`,
    };
}

// ---------- player card ----------
function pcard(card, opts = {}) {
    // Card colour is ALWAYS the higher of the two OVRs, so the same player is
    // the same colour on every device / view (batting, bowling, auction, XI).
    const relevantOvr = Math.max(card.batting_ovr || 0, card.bowling_ovr || 0);
    const cls = ['pcard', tierClass(relevantOvr)];
    if (opts.out) cls.push('out');
    if (opts.disabled) cls.push('disabled');
    if (opts.selectable) cls.push('selectable');
    if (opts.selected) cls.push('selected');
    const fig = opts.figure ? `<div class="pfig">${opts.figure}</div>` : '';
    const tag = opts.tag ? `<div class="ptag">${opts.tag}</div>` : '';
    const jersey = jerseyAttrs(opts.jerseyColor, opts.jerseyStyle);
    // flip-to-back (role + 3x3 phase stats). Flipped state is kept in a global
    // set keyed by name so it survives the constant re-renders (see the
    // capture-phase handler below); the button stops the click from also
    // triggering the card's own select/drag action.
    const canFlip = !!card.style_fit;
    if (canFlip) cardBackData.set(card.name, card);   // for the Bat/Bowl tab re-render
    if (canFlip && flippedCards.has(card.name)) cls.push('flipped');
    const flipBtn = canFlip ? `<button type="button" class="pcard-flip" title="Role &amp; phase stats">i</button>` : '';
    const back = canFlip ? pcardBack(card) : '';
    // at most one role badge (the single best cell >=70, see
    // compile_player_stats.py) -- nothing shown if the player didn't clear
    // the threshold anywhere, which is the common/correct case for bowlers.
    const roleBadge = (card.roles && card.roles.length)
        ? `<div class="prole"><span>${card.roles[0].label} <b>${card.roles[0].score}</b></span></div>`
        : '';
    return `
    <div class="${cls.join(' ')}" data-card-name="${card.name}" ${opts.attrs || ''}${jersey.attr} style="${jersey.style}">
        ${opts.extraHtml || ''}
        ${flipBtn}
        <div class="pcard-face pcard-front">
            <div class="pcard-top">
                <div class="ovr-chip"><b>${card.batting_ovr ?? '--'}</b><span>BAT</span></div>
                <div class="ovr-chip"><b>${card.bowling_ovr ?? '--'}</b><span>BOWL</span></div>
            </div>
            <div class="pcard-body">
                <div class="pname">${card.name}</div>
                ${fig}${tag}${roleBadge}
            </div>
        </div>
        ${back}
    </div>`;
}


const flippedCards = new Set();   // card names currently showing their back
const cardBackTab = new Map();    // card name -> 'bat' | 'bowl' (default 'bat')

function pcardBack(card) {
    const cell = (v) => {
        const lvl = v >= 80 ? 'hi' : v >= 55 ? 'mid' : 'lo';
        return `<td class="fit-${lvl}">${v}</td>`;
    };
    const grid = (fit, cols) => {
        const row = (label, ph) => `<tr><th>${label}</th>${cols.map(c => cell(fit[ph][c.key])).join('')}</tr>`;
        return `<table class="fit-grid">
            <tr><th></th>${cols.map(c => `<th>${c.hdr}</th>`).join('')}</tr>
            ${row('PP', 'pp')}${row('MID', 'mid')}${row('DTH', 'death')}
        </table>`;
    };
    const hasBowl = !!card.bowl_fit;
    const tab = hasBowl ? (cardBackTab.get(card.name) || 'bat') : 'bat';
    // both phase grids: batting (attack/defence/rotate) and bowling (attack/contain/defend) --
    // "defence" reads the same underlying `anchor` data field (survival/balls-per-dismissal);
    // the label is renamed to match the Defend role button, the data key is unchanged.
    const batGrid = grid(card.style_fit, [
        { key: 'attack', hdr: 'ATK' }, { key: 'anchor', hdr: 'DEF' }, { key: 'rotate', hdr: 'ROT' }]);
    const bowlGrid = hasBowl ? grid(card.bowl_fit, [
        { key: 'attack', hdr: 'ATK' }, { key: 'contain', hdr: 'CON' }, { key: 'defend', hdr: 'DEF' }]) : '';
    const tabs = hasBowl
        ? `<div class="pcard-tabs">
            <button type="button" class="pcard-tab${tab === 'bat' ? ' on' : ''}" data-backtab="bat">BAT</button>
            <button type="button" class="pcard-tab${tab === 'bowl' ? ' on' : ''}" data-backtab="bowl">BOWL</button>
        </div>` : '';
    // at most one role label now (see compile_player_stats.py) -- nothing shown
    // at all if the player didn't clear the 70 threshold in any cell
    const roleList = (card.roles && card.roles.length)
        ? `${card.roles[0].label} ${card.roles[0].score}`
        : '';
    return `
    <div class="pcard-face pcard-back">
        <div class="pcard-back-name">${card.name}</div>
        ${tabs}
        ${tab === 'bowl'
            ? `<div class="pcard-role pcard-role-bowl">Bowling phases</div>${bowlGrid}`
            : `<div class="pcard-role">${roleList}</div>${batGrid}`}
    </div>`;
}

// Bat/Bowl tab switch on the card back — capture phase for the same reason as
// the flip button (don't let the tap select/drag the card underneath).
document.addEventListener('click', (e) => {
    const tabBtn = e.target.closest('.pcard-tab');
    if (!tabBtn) return;
    e.stopPropagation();
    e.preventDefault();
    const cardEl = tabBtn.closest('.pcard');
    const name = cardEl && cardEl.dataset.cardName;
    if (!name) return;
    cardBackTab.set(name, tabBtn.dataset.backtab);
    // re-render just this card's back face in place
    const backEl = cardEl.querySelector('.pcard-back');
    if (backEl) {
        const tmp = document.createElement('div');
        tmp.innerHTML = pcardBack(cardForBack(cardEl, name));
        backEl.replaceWith(tmp.firstElementChild);
    }
}, true);

// The click handler above needs the card's fit data at tap time, but all we
// have is the DOM node — so cards stash their data on the element when built.
const cardBackData = new Map();   // name -> {style_fit, bowl_fit, roles, role}
function cardForBack(cardEl, name) {
    return cardBackData.get(name) || { name, style_fit: null, bowl_fit: null };
}

// Toggling on pointerdown (not click) makes the flip register on the very
// first tap instead of needing 2-3 tries: a "click" only fires once the
// browser can match a mouseup/touchend to the SAME element the mousedown/
// touchstart started on, and the ~700ms state-poll re-renders the whole
// card grid on every version bump -- if that re-render happens to land
// between press and release (entirely plausible with a slower or hesitant
// tap), the original button is gone and no click ever fires. pointerdown
// needs only the single initial press, so it can't be interrupted that way.
function handleFlipToggle(e) {
    const btn = e.target.closest('.pcard-flip');
    if (!btn) return;
    e.stopPropagation();
    e.preventDefault();
    const cardEl = btn.closest('.pcard');
    const name = cardEl && cardEl.dataset.cardName;
    if (!name) return;
    if (flippedCards.has(name)) { flippedCards.delete(name); cardEl.classList.remove('flipped'); }
    else { flippedCards.add(name); cardEl.classList.add('flipped'); }
}
document.addEventListener('pointerdown', handleFlipToggle, true);
// pointerdown's stopPropagation() doesn't cancel the click that still
// follows it -- swallow that separately so it never reaches the card's own
// select/drag click handler (tapping flip on a bench card shouldn't also
// try to select that bowler/batsman).
document.addEventListener('click', (e) => {
    if (e.target.closest('.pcard-flip')) { e.stopPropagation(); e.preventDefault(); }
}, true);

function emptySlot(text, dropAttrs = '') {
    return `<div class="pcard is-empty" ${dropAttrs}>${text}</div>`;
}

// Role picker (replaces the intent slider). The player's 0-99 grid grade for
// each role stays hidden — the engine uses it (Stage 5 bonus), the buttons
// deliberately don't show it (players read skill off the card's flip side).
function roleButtons(kind, selected, disabled, fit, phKey, defs, batterName) {
    const batterAttr = batterName ? ` data-rolebatter="${batterName}"` : '';
    const btns = defs.map(d => {
        const on = d.key === selected ? ' on' : '';
        return `<button type="button" class="role-btn${on}" data-role="${d.key}" data-rolekind="${kind}"${batterAttr}
            style="--racc:${d.color}" ${disabled ? 'disabled' : ''}>
            <span class="role-lbl">${d.label}</span></button>`;
    }).join('');
    return `<div class="role-picker">${btns}</div>`;
}

// Small floating "?" popover explaining the roles in plain english. Absolutely
// positioned inside #ground so it never takes layout space from the cards.
// Built from the same defs (label + color) the buttons use, so the popover
// and the buttons always agree.
const ROLE_HELP_TEXT = {
    attack_bat: 'Swing hard — more 4s & 6s, but a bigger chance of getting out.',
    rotate: 'Work the gaps — more 1s & 2s, giving up the big shots.',
    defend_bat: 'Block — much safer, but the scoring dries up.',
    attack_bowl: 'Hunt the wicket — best chance of a wicket, but you leak runs.',
    contain: 'Bring the fielders in — 1s & 2s dry up, but batsmen may clear the ring for boundaries.',
    defend_bowl: 'Guard the boundary — 4s & 6s dry up, but easy singles are on offer.',
};
function roleHelpHtml(iAmBatting) {
    const defs = iAmBatting ? BAT_ROLE_DEFS : BOWL_ROLE_DEFS;
    const suffix = iAmBatting ? '_bat' : '_bowl';
    const rows = defs.map(d => {
        const text = ROLE_HELP_TEXT[d.key] || ROLE_HELP_TEXT[d.key + suffix];
        return `<div class="rh-row"><b style="--racc:${d.color}">${d.label}</b><span>${text}</span></div>`;
    }).join('');
    return `<button type="button" id="role-help-btn" title="What do the roles do?">?</button>
        <div id="role-help-panel" class="${ui.roleHelpOpen ? 'open' : ''}">
            <div class="rh-title">${iAmBatting ? 'Batting roles' : 'Bowling roles'}</div>${rows}</div>`;
}

function wireRoleHelp() {
    const btn = document.getElementById('role-help-btn');
    const panel = document.getElementById('role-help-panel');
    if (!btn || !panel) return;
    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        ui.roleHelpOpen = !ui.roleHelpOpen;
        panel.classList.toggle('open', ui.roleHelpOpen);
    });
}

function batRoleButtons(m, kind, batter, disabled) {
    return roleButtons(kind, getBatterRole(batter.name), disabled, batter.style_fit,
                       phaseKey(m), BAT_ROLE_DEFS, batter.name);
}
function bowlRoleButtons(m, fit, disabled) {
    return roleButtons('bowl', ui.bowlRole, disabled, fit, phaseKey(m), BOWL_ROLE_DEFS);
}

// ---------- ground ----------
function bowlerCard(cb, jerseyColor, jerseyStyle) {
    return pcard({ name: cb.name, batting_ovr: cb.batting_ovr ?? null, bowling_ovr: cb.bowling_ovr,
                   role: cb.role, roles: cb.roles, style_fit: cb.style_fit, bowl_fit: cb.bowl_fit },
        { figure: `${cb.wickets}-${cb.runs}(${cb.overs})`, jerseyColor, jerseyStyle });
}

function renderGround(m) {
    const g = $('ground');
    const isFreeHit = m.stage === 'free_hit';
    const isResume = m.stage === 'await_resume';
    const locked = isFreeHit ? m.free_hit.i_ready : isResume ? (m.resume && m.resume.i_ready) : m.pending.i_submitted;
    const canIntent = m.stage === 'play' || m.stage === 'free_hit' || m.stage === 'await_resume';
    const slDisabled = locked || !canIntent;
    // retiring is only allowed between overs, before either half is submitted
    const canRetire = m.i_am_batting && m.stage === 'play' && !m.pending.i_submitted && !m.pending.opponent_submitted;
    const retireBtn = (which) => canRetire
        ? `<button class="btn-retire" data-retire="${which}">Retire Hurt</button>` : '';

    // --- batsmen (top row) ---
    let strikerSlot;
    if (m.striker) {
        strikerSlot = `<div class="ground-slot">
            <div class="slot-label">On Strike</div>
            ${pcard(m.striker, { figure: `${m.striker.runs}(${m.striker.balls})`,
                                  jerseyColor: m.batting_team_color, jerseyStyle: m.batting_team_jersey })}
            ${m.i_am_batting ? batRoleButtons(m, 'striker', m.striker, slDisabled) : ''}
            ${retireBtn('striker')}
        </div>`;
    } else if (m.await_next_batter && m.i_am_batting) {
        strikerSlot = `<div class="ground-slot">
            <div class="slot-label">New Batsman (below)</div>
            ${emptySlot('Drop a batsman here', 'id="drop-striker" data-drop="1"')}
        </div>`;
    } else {
        strikerSlot = `<div class="ground-slot"><div class="slot-label">On Strike</div>${emptySlot('—')}</div>`;
    }

    let nonStrikerSlot;
    if (m.non_striker) {
        nonStrikerSlot = `<div class="ground-slot">
            <div class="slot-label">Non-Striker</div>
            ${pcard(m.non_striker, { figure: `${m.non_striker.runs}(${m.non_striker.balls})`,
                                      jerseyColor: m.batting_team_color, jerseyStyle: m.batting_team_jersey })}
            ${m.i_am_batting ? batRoleButtons(m, 'nonstriker', m.non_striker, slDisabled) : ''}
            ${retireBtn('non_striker')}
        </div>`;
    } else {
        nonStrikerSlot = `<div class="ground-slot"><div class="slot-label">Non-Striker</div>${emptySlot('—')}</div>`;
    }

    // --- bowler (bottom row) ---
    let bowlerSlot;
    if (m.i_am_batting) {
        const ob = m.pending && m.pending.opponent_bowler;   // revealed once bowler locks
        let label, cardHtml;
        if (ob) { label = 'Bowler This Over'; cardHtml = bowlerCard(ob, m.bowling_team_color, m.bowling_team_jersey); }
        else if (m.current_bowler) { label = "Last Over's Bowler"; cardHtml = bowlerCard(m.current_bowler, m.bowling_team_color, m.bowling_team_jersey); }
        else { label = 'Bowler'; cardHtml = emptySlot('Awaiting bowler…'); }
        bowlerSlot = `<div class="ground-slot"><div class="slot-label">${label}</div>${cardHtml}</div>`;
    } else if (m.stage === 'play') {
        const sel = ui.selectedBowler ? m.my_bench.find(b => b.name === ui.selectedBowler) : null;
        bowlerSlot = `<div class="ground-slot"><div class="slot-label">Your Bowler This Over</div>` +
            (sel
                ? pcard(sel, { tag: `${sel.overs_bowled}/${sel.max_overs} overs`,
                               jerseyColor: m.bowling_team_color, jerseyStyle: m.bowling_team_jersey }) +
                bowlRoleButtons(m, sel.bowl_fit, slDisabled)
                : emptySlot('Pick a bowler from your bench (below)')) +
            `</div>`;
    } else {
        bowlerSlot = `<div class="ground-slot"><div class="slot-label">Your Bowler</div>` +
            (m.current_bowler ? bowlerCard(m.current_bowler, m.bowling_team_color, m.bowling_team_jersey) : emptySlot('—')) +
            (isFreeHit ? bowlRoleButtons(m, m.current_bowler && m.current_bowler.bowl_fit, slDisabled) : '') +
            `</div>`;
    }

    g.innerHTML = `<div class="ground-row batters-row">${strikerSlot}${nonStrikerSlot}</div>
                   ${farmToggle(m, slDisabled)}
                   <div class="ground-row bowler-row">${bowlerSlot}</div>`;
    wireRoleButtons();
    wireFarmToggle();
    wireDropZone(m);
    wireRetireButtons();
}

// Strike farming: only offered to the batting side, and only when the pair at
// the crease is mismatched enough for it to mean anything (the server decides
// that via FARM_MIN_GAP and sends can_farm).
function farmToggle(m, disabled) {
    if (!m.i_am_batting || !m.can_farm) return '';
    const on = ui.farmStrike;
    return `<div class="farm-row ${on ? 'on' : ''} ${disabled ? 'is-disabled' : ''}" id="farm-toggle">
        <div class="farm-label">
            <span class="farm-title">Farm the strike</span>
            <span class="farm-hint">${on
                ? 'Your better batsman keeps the strike early, then works a single to hold it for the next over.'
                : 'Protect the weaker batsman by turning down singles early in the over.'}</span>
        </div>
        <div class="switch"></div>
    </div>`;
}

function wireFarmToggle() {
    const el = $('farm-toggle');
    if (!el || el.classList.contains('is-disabled')) return;
    el.addEventListener('click', () => {
        ui.farmStrike = !ui.farmStrike;
        el.classList.toggle('on', ui.farmStrike);
        const hint = el.querySelector('.farm-hint');
        if (hint) {
            hint.textContent = ui.farmStrike
                ? 'Your better batsman keeps the strike early, then works a single to hold it for the next over.'
                : 'Protect the weaker batsman by turning down singles early in the over.';
        }
        if (navigator.vibrate) navigator.vibrate(8);
    });
}

function wireRetireButtons() {
    document.querySelectorAll('#ground [data-retire]').forEach(btn => {
        btn.addEventListener('click', () => retireBatsman(btn.dataset.retire));
    });
}

async function retireBatsman(which) {
    const label = which === 'striker' ? (CURRENT.match.striker && CURRENT.match.striker.name)
        : (CURRENT.match.non_striker && CURRENT.match.non_striker.name);
    if (!confirm(`Retire ${label || 'this batsman'}? They will NOT be able to return to bat.`)) return;
    try {
        await Net.post('/api/retire_batsman', { token: Net.getToken(), which });
        Net.forceRefresh();
    } catch (e) { toast(e.message); }
}

function wireRoleButtons() {
    document.querySelectorAll('#ground .role-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            if (btn.disabled) return;
            const role = btn.dataset.role, kind = btn.dataset.rolekind, bn = btn.dataset.rolebatter;
            if (kind === 'bowl') ui.bowlRole = role;
            else if (bn) setBatterRole(bn, role);
            // reflect the selection within this picker without a full re-render
            btn.parentElement.querySelectorAll('.role-btn').forEach(b => b.classList.toggle('on', b === btn));
            if (navigator.vibrate) navigator.vibrate(8);
        });
    });
}

// ---------- ready bar ----------
function oppReadyTxt(ready) {
    return `<span class="ready-status">Opponent: ${ready
        ? '<span class="on">READY</span>' : '<span class="off">setting up…</span>'}</span>`;
}

function renderReadyBar(m) {
    const bar = $('ready-bar');

    if (m.stage === 'await_batter') {
        bar.innerHTML = m.i_am_batting
            ? `<div class="ready-status"><span class="off">Send in a new batsman — tap or drag one from your bench.</span></div>${impactButtonHtml(m)}`
            : `<div class="ready-status"><span class="off">Waiting for the batting side to send a new batsman…</span></div>`;
        if (m.i_am_batting) wireImpactButton();
        return;
    }

    if (m.stage === 'await_resume') {
        const oppTxt = oppReadyTxt(m.resume && m.resume.opponent_ready);
        if (m.resume && m.resume.i_ready) {
            bar.innerHTML = `<button class="btn-ghost" disabled>Ready — waiting…</button>${oppTxt}`;
        } else if (m.i_am_batting) {
            bar.innerHTML = `${impactButtonHtml(m)}
                <button class="btn-go btn-lg" id="btn-resume">Ready to Resume Over</button>
                <span class="ready-status"><span class="off">New batsman is in — pick a role, then resume.</span></span>${oppTxt}`;
            $('btn-resume').addEventListener('click', submitResume);
            wireImpactButton();
        } else {
            bar.innerHTML = `${impactButtonHtml(m)}
                <button class="btn-go btn-lg" id="btn-resume">Ready to Resume Over</button>
                <span class="ready-status"><span class="off">New batsman is in — react with your bowling role, then resume.</span></span>${oppTxt}`;
            $('btn-resume').addEventListener('click', submitResume);
            wireImpactButton();
        }
        return;
    }

    if (m.stage === 'free_hit') {
        const oppTxt = oppReadyTxt(m.free_hit.opponent_ready);
        if (m.free_hit.i_ready) {
            bar.innerHTML = `<button class="btn-ghost" disabled>Free hit locked — waiting…</button>${oppTxt}`;
        } else {
            bar.innerHTML = `<span class="ready-status" style="color:var(--gold)">FREE HIT!</span>
                <button class="btn-gold btn-lg" id="btn-fh">Confirm Intent</button>${oppTxt}`;
            $('btn-fh').addEventListener('click', submitFreeHit);
        }
        return;
    }

    // stage === 'play' — SEQUENCED: the bowling side locks the bowler first,
    // then the batting side sees who's bowling and sets its strategy.
    if (!m.i_am_batting) {
        if (m.pending.i_submitted) {
            bar.innerHTML = `<button class="btn-ghost" disabled>Bowler locked — waiting for the batsmen…</button>`;
        } else {
            const disabled = !ui.selectedBowler;
            bar.innerHTML =
                `${gambitToggleHtml(m, 'trap', 'Set Trap')}
                 ${impactButtonHtml(m)}
                 <button class="btn-go btn-lg" id="btn-ready" ${disabled ? 'disabled' : ''}>Lock In Bowler</button>
                 <span class="ready-status"><span class="off">${disabled ? 'Pick a bowler first' : "Batsmen won't see your role"}</span></span>`;
            const btn = $('btn-ready');
            if (btn) btn.addEventListener('click', submitOver);
            wireGambitToggle();
            wireImpactButton();
        }
        return;
    }

    // batting side
    if (m.pending.i_submitted) {
        bar.innerHTML = `<button class="btn-ghost" disabled>Locked in — playing the over…</button>`;
        return;
    }
    if (!m.pending.opponent_submitted) {
        bar.innerHTML = `<div class="ready-status"><span class="off">Waiting for the bowling side to lock in the bowler…</span></div>`;
        return;
    }
    const bn = m.pending.opponent_bowler ? m.pending.opponent_bowler.name : 'the bowler';
    bar.innerHTML =
        `<span class="ready-status" style="color:var(--gold)">Bowling: ${bn}</span>
         ${gambitToggleHtml(m, 'attack', 'All Out Attack')}
         ${impactButtonHtml(m)}
         <button class="btn-go btn-lg" id="btn-ready">Set Strategy &amp; Ready</button>`;
    $('btn-ready').addEventListener('click', submitOver);
    wireGambitToggle();
    wireImpactButton();
}

// one-shot gambit cards: armed locally, sent (secretly) with the submission
const GAMBIT_EXPLAIN = {
    trap: "One-shot: this over, the striker's wicket chance climbs (hardest if they're batting aggressively, wasted on a blocker) and their 4/6 chance drops. 1s, 2s and dot-ball chance are untouched.",
    attack: "One-shot: this over, your 4 and 6 chances go up and your wicket risk drops. 1s, 2s and dot-ball chance are untouched.",
};
function gambitToggleHtml(m, kind, label) {
    const avail = m.gambits && m.gambits.available && m.gambits.available[kind];
    const hint = `<div class="gambit-hint">${GAMBIT_EXPLAIN[kind]}</div>`;
    if (!avail) return `<span class="gambit-btn used" title="${GAMBIT_EXPLAIN[kind]}">${label} — used</span>${hint}`;
    return `<button class="gambit-btn${ui.armGambit ? ' armed' : ''}" id="btn-gambit" title="${GAMBIT_EXPLAIN[kind]}">${label}${ui.armGambit ? ' — ARMED' : ''}</button>${hint}`;
}

function wireGambitToggle() {
    const b = $('btn-gambit');
    if (b) b.addEventListener('click', () => {
        ui.armGambit = !ui.armGambit;
        if (CURRENT) renderGame(CURRENT);
    });
}

// ---------- Impact Player: one bench-for-XI swap per team, once per match ----------
function impactButtonHtml(m) {
    if (!m.impact) return '';
    if (m.impact.used) return `<span class="gambit-btn used" title="${m.impact.swap_text || 'Already used this match'}">Impact Player — used</span>`;
    if (!m.impact.can_use) return '';
    return `<button class="gambit-btn" id="btn-impact">Impact Player</button>`;
}

function wireImpactButton() {
    const b = $('btn-impact');
    if (b) b.addEventListener('click', openImpactOverlay);
}

function openImpactOverlay() {
    ui.impactPick = { in: null, out: null };
    renderImpactOverlay(CURRENT.match);
    $('impact-overlay').classList.remove('hidden');
}

function closeImpactOverlay() {
    $('impact-overlay').classList.add('hidden');
}

function renderImpactOverlay(m) {
    const imp = m.impact || { pool: [], out_options: [] };
    const myColor = m.i_am_batting ? m.batting_team_color : m.bowling_team_color;
    const myJersey = m.i_am_batting ? m.batting_team_jersey : m.bowling_team_jersey;
    const inCards = imp.pool.map(p => pcard(p, {
        ovr: Math.max(p.batting_ovr, p.bowling_ovr), selectable: true,
        selected: ui.impactPick.in === p.name, attrs: `data-impact-in="${p.name}"`,
        jerseyColor: myColor, jerseyStyle: myJersey,
    })).join('') || '<div class="bench-empty">No bench players available.</div>';
    const outCards = imp.out_options.map(p => pcard(p, {
        ovr: Math.max(p.batting_ovr, p.bowling_ovr), selectable: true, out: p.dismissed,
        selected: ui.impactPick.out === p.name, attrs: `data-impact-out="${p.name}"`,
        jerseyColor: myColor, jerseyStyle: myJersey,
    })).join('') || '<div class="bench-empty">No one eligible to replace right now.</div>';
    const ready = ui.impactPick.in && ui.impactPick.out;
    $('impact-card').innerHTML = `
        <h2>Bring On Your Impact Player</h2>
        <div class="sub">One-shot for the whole match — pick who comes IN, then who they replace.</div>
        <h4 style="margin-top:1rem;">Bring In</h4>
        <div class="xi-list">${inCards}</div>
        <h4 style="margin-top:1rem;">Replaces</h4>
        <div class="xi-list">${outCards}</div>
        <div style="display:flex; margin-top:1rem; justify-content:center; gap:0.8rem; flex-wrap:wrap;">
            <button class="btn-go btn-lg" id="impact-confirm" ${ready ? '' : 'disabled'}>Confirm Swap</button>
            <button class="btn-ghost" id="impact-cancel">Cancel</button>
        </div>`;
    $('impact-card').querySelectorAll('[data-impact-in]').forEach(el =>
        el.addEventListener('click', () => { ui.impactPick.in = el.dataset.impactIn; renderImpactOverlay(m); }));
    $('impact-card').querySelectorAll('[data-impact-out]').forEach(el =>
        el.addEventListener('click', () => { ui.impactPick.out = el.dataset.impactOut; renderImpactOverlay(m); }));
    const confirmBtn = $('impact-confirm');
    if (confirmBtn) confirmBtn.addEventListener('click', confirmImpactSub);
    $('impact-cancel').addEventListener('click', closeImpactOverlay);
}

async function confirmImpactSub() {
    if (!ui.impactPick.in || !ui.impactPick.out) return;
    try {
        await Net.post('/api/impact_sub', {
            token: Net.getToken(), in_name: ui.impactPick.in, out_name: ui.impactPick.out,
        });
        closeImpactOverlay();
        Net.forceRefresh();
    } catch (e) { toast(e.message); }
}

async function submitOver() {
    const m = CURRENT.match;
    try {
        if (m.i_am_batting) {
            await Net.post('/api/submit_over', {
                token: Net.getToken(),
                striker_role: getBatterRole(m.striker.name),
                non_striker_role: getBatterRole(m.non_striker.name),
                gambit: ui.armGambit,
                farm_strike: ui.farmStrike,
            });
        } else {
            if (!ui.selectedBowler) return toast('Pick a bowler first.');
            await Net.post('/api/submit_over', {
                token: Net.getToken(),
                bowler_name: ui.selectedBowler,
                bowl_role: ui.bowlRole,
                gambit: ui.armGambit,
            });
        }
        ui.armGambit = false;
        Net.forceRefresh();
    } catch (e) { toast(e.message); }
}

async function submitFreeHit() {
    const m = CURRENT.match;
    const body = m.i_am_batting
        ? { token: Net.getToken(), striker_role: getBatterRole(m.striker.name), non_striker_role: getBatterRole(m.non_striker.name) }
        : { token: Net.getToken(), bowl_role: ui.bowlRole };
    try { await Net.post('/api/free_hit', body); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
}

async function submitResume() {
    const m = CURRENT.match;
    const body = m.i_am_batting
        ? { token: Net.getToken(), striker_role: getBatterRole(m.striker.name),
            non_striker_role: getBatterRole(m.non_striker.name), farm_strike: ui.farmStrike }
        : { token: Net.getToken(), bowl_role: ui.bowlRole };
    try { await Net.post('/api/ready_resume', body); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
}

// ---------- bench ----------
function renderBench(m) {
    const bench = $('bench');
    const title = $('bench-title');
    let html = '';

    if (m.i_am_batting) {
        title.textContent = m.await_next_batter ? 'Your Bench — tap or drag a batsman in' : 'Your Bench';
        const avail = m.my_bench;
        if (!avail.length) { bench.innerHTML = '<div class="bench-empty">No batsmen on the bench.</div>'; return; }
        avail.forEach(b => {
            const out = b.status === 'out';
            const canPick = m.await_next_batter && b.status === 'available';
            html += pcard(b, {
                ovr: b.batting_ovr,
                out,
                selectable: canPick,
                attrs: canPick ? `draggable="true" data-batter="${b.name}"` : '',
                tag: out ? 'OUT' : (b.status === 'available' ? 'Ready' : ''),
                jerseyColor: m.batting_team_color, jerseyStyle: m.batting_team_jersey,
            });
        });
    } else {
        title.textContent = 'Your Bowlers';
        m.my_bench.forEach(b => {
            const selected = ui.selectedBowler === b.name;
            html += pcard(b, {
                ovr: b.bowling_ovr,
                disabled: b.disabled,
                selectable: !b.disabled,
                selected,
                attrs: !b.disabled ? `data-bowler="${b.name}"` : '',
                tag: `${b.overs_bowled}/${b.max_overs} overs${b.disabled ? ' • rest' : ''}`,
                jerseyColor: m.bowling_team_color, jerseyStyle: m.bowling_team_jersey,
            });
        });
    }
    bench.innerHTML = html;
    wireBench(m);
}

function wireBench(m) {
    if (m.i_am_batting) {
        document.querySelectorAll('#bench [data-batter]').forEach(el => {
            el.addEventListener('click', () => sendNextBatter(el.dataset.batter));
            el.addEventListener('dragstart', (ev) => {
                ev.dataTransfer.setData('text/plain', el.dataset.batter);
                el.classList.add('dragging');
            });
            el.addEventListener('dragend', () => el.classList.remove('dragging'));
        });
    } else {
        document.querySelectorAll('#bench [data-bowler]').forEach(el => {
            el.addEventListener('click', () => {
                if (CURRENT.match.stage !== 'play' || CURRENT.match.pending.i_submitted) return;
                ui.selectedBowler = el.dataset.bowler;
                renderGround(CURRENT.match);
                renderReadyBar(CURRENT.match);
                renderBench(CURRENT.match);
            });
        });
    }
}

function wireDropZone(m) {
    const zone = $('drop-striker');
    if (!zone) return;
    zone.addEventListener('dragover', (ev) => { ev.preventDefault(); zone.classList.add('drop-hot'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drop-hot'));
    zone.addEventListener('drop', (ev) => {
        ev.preventDefault();
        const name = ev.dataTransfer.getData('text/plain');
        if (name) sendNextBatter(name);
    });
}

async function sendNextBatter(name) {
    try {
        await Net.post('/api/set_next_batter', { token: Net.getToken(), batter_name: name });
        Net.forceRefresh();
    } catch (e) { toast(e.message); }
}

// ---------- commentary (this over) ----------
function commLine(entry) {
    const cls = ['comm-line'];
    if (entry.type === 'boundary') cls.push('boundary');
    else if (entry.type === 'wicket') cls.push('wicket');
    else if (entry.type === 'extra') cls.push('extra');
    else if (entry.type === 'milestone') cls.push('milestone');
    const ball = entry.ball ? `<span class="bl">${entry.ball}</span>` : '';
    // highlighted outcome box at the start of the line
    let badge = '';
    if (entry.outcome) {
        let mod = '';
        if (entry.type === 'boundary') mod = ' b';
        else if (entry.type === 'wicket') mod = ' w';
        else if (entry.type === 'extra') mod = ' e';
        badge = `<span class="oc${mod}">${entry.outcome}</span>`;
    }
    return `<div class="${cls.join(' ')}">${ball}${badge}${entry.text}</div>`;
}

// Per-entry reveal weight: dots/singles stay brisk so overs don't drag, the
// big moments (boundaries, wickets) get a beat of extra weight to land.
function ballRevealDelay(entry) {
    if (entry.type === 'milestone') return 150;
    if (entry.type === 'wicket') return 520;
    if (entry.type === 'boundary') return entry.outcome === '6' ? 540 : 460;
    if (entry.type === 'extra') return 280;
    if (entry.outcome === '0') return 240;
    return 290; // 1, 2, 3, 5
}

// lightweight schematic ball-flight arc, keyed off the outcome only
const TRAJECTORY_ARCS = {
    '0': { d: 'M 50 96 Q 50 90 50 84', cls: 'traj-run' },
    '1': { d: 'M 50 96 Q 62 72 74 52', cls: 'traj-run' },
    '2': { d: 'M 50 96 Q 66 58 84 32', cls: 'traj-run' },
    '3': { d: 'M 50 96 Q 38 56 24 30', cls: 'traj-run' },
    '5': { d: 'M 50 96 Q 32 54 14 26', cls: 'traj-run' },
    '4': { d: 'M 50 96 Q 78 44 98 10', cls: 'traj-four' },
    '6': { d: 'M 50 96 Q 58 6 48 -6', cls: 'traj-six' },
};
function trajectorySvg(entry) {
    // outcome keys are plain run counts ('0'-'6') -- wicket/extra/milestone
    // entries use non-numeric outcome codes (Out/Wd/Nb/HT/etc) and simply
    // won't match, so no separate type check is needed here.
    const arc = TRAJECTORY_ARCS[entry.outcome];
    if (!arc) return '';
    return `<svg class="ball-traj ${arc.cls}" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path d="${arc.d}"/></svg>`;
}

function showBallStamp(entry) {
    const overlay = $('ball-stamp-overlay');
    if (!overlay) { playOutcomeSound(entry.outcome); return; }
    let cls = 'run', label = '+' + entry.outcome;
    if (entry.type === 'boundary') { cls = entry.outcome === '6' ? 'six' : 'four'; label = entry.outcome === '6' ? 'SIX!' : 'FOUR!'; }
    else if (entry.type === 'wicket') { cls = 'wicket'; label = 'OUT!'; }
    else if (entry.type === 'extra') { cls = 'extra'; label = entry.outcome === 'Wd' ? 'WIDE' : 'NO BALL'; }
    else if (entry.outcome === '0') { cls = 'dot'; label = '•'; }
    overlay.innerHTML = `${trajectorySvg(entry)}<div class="ball-stamp ${cls}">${label}</div>`;
    overlay.classList.remove('hidden');
    setTimeout(() => overlay.classList.add('hidden'), 560);

    if (entry.type === 'wicket' || entry.outcome === '6') {
        const wrap = document.querySelector('.ground-wrap');
        if (wrap) { wrap.classList.remove('screen-shake'); void wrap.offsetWidth; wrap.classList.add('screen-shake'); }
    }
    if (entry.type === 'wicket') {
        const flash = document.createElement('div');
        flash.className = 'screen-flash';
        document.body.appendChild(flash);
        setTimeout(() => flash.remove(), 400);
    }
    playOutcomeSound(entry.outcome);
}

// Only 50/century/hat-trick are "big" milestones worth a full-screen toast --
// server-side, plain type:'milestone' commentary also covers end-of-over
// summaries, end-of-innings, and gambit reveals (all outcome:null), which
// should stay as ordinary log lines, not pop the toast every single over.
function isBigMilestone(entry) {
    return entry.type === 'milestone' && ['50', '100', 'HT'].includes(entry.outcome);
}

function showMilestoneToast(entry) {
    const root = $('milestone-toast-root');
    if (!root) return;
    const title = entry.outcome === 'HT' ? 'HAT-TRICK!' : entry.outcome === '100' ? 'CENTURY!' : 'FIFTY!';
    const el = document.createElement('div');
    el.className = 'milestone-toast';
    el.innerHTML = `<div class="mt-title">${title}</div><div class="mt-sub">${entry.text.replace(/^[^\s]+\s/, '')}</div>`;
    root.appendChild(el);
    playOutcomeSound(entry.outcome);
    setTimeout(() => el.remove(), 2800);
}

function renderThisOver(m) {
    const box = $('this-over');
    const entries = m.this_over;
    if (!entries.length) {
        box.innerHTML = '<div class="comm-empty">The over will play out here, ball by ball.</div>';
        overAnim.key = null;
        overAnim.shown = 0;
        overAnim.timeouts.forEach(clearTimeout);
        overAnim.timeouts = [];
        $('btn-skip-reveal').classList.add('hidden');
        return;
    }
    // key changes when a fresh over starts (server clears this_over each over)
    const key = m.innings + ':' + entries[0].ball;
    if (key !== overAnim.key) {
        overAnim.key = key;
        overAnim.shown = 0;
        overAnim.timeouts.forEach(clearTimeout);
        overAnim.timeouts = [];
        box.innerHTML = '';
    }
    // reveal only the not-yet-shown deliveries, one after another, each
    // carrying its own weighted delay (see ballRevealDelay)
    const startIdx = overAnim.shown;
    const newCount = entries.length - startIdx;
    if (newCount > 0) {
        let cumDelay = 0;
        for (let i = startIdx; i < entries.length; i++) {
            const entry = entries[i];
            const delay = cumDelay;
            cumDelay += ballRevealDelay(entry);
            const tid = setTimeout(() => {
                box.insertAdjacentHTML('beforeend', commLine(entry));
                box.scrollTop = box.scrollHeight;
                if (isBigMilestone(entry)) showMilestoneToast(entry);
                else if (entry.type !== 'milestone') showBallStamp(entry);
            }, delay);
            overAnim.timeouts.push(tid);
        }
        // so the caller can hold off popping the result banner over an
        // in-progress reveal (see render()'s `finished` handling)
        overAnim.revealUntil = Date.now() + cumDelay + 200;
        $('btn-skip-reveal').classList.remove('hidden');
        setTimeout(() => $('btn-skip-reveal').classList.add('hidden'), cumDelay + 50);
    }
    overAnim.shown = entries.length;
}

function skipReveal() {
    overAnim.timeouts.forEach(clearTimeout);
    overAnim.timeouts = [];
    if (CURRENT && CURRENT.match) {
        $('this-over').innerHTML = CURRENT.match.this_over.map(commLine).join('');
        $('this-over').scrollTop = $('this-over').scrollHeight;
    }
    $('ball-stamp-overlay').classList.add('hidden');
    $('btn-skip-reveal').classList.add('hidden');
    overAnim.revealUntil = 0;
}

// ---------- opponent list ----------
function renderOppList(m) {
    $('opp-head').textContent = m.i_am_batting ? 'Opposition Bowlers' : 'Opposition Batsmen';
    $('opp-list').innerHTML = m.opponent_list.map(p =>
        `<div class="opp-row"><span>${p.name}</span>
         <span class="ovrs">${p.batting_ovr}/${p.bowling_ovr}</span></div>`).join('');
}

// ---------- live tab ----------
function renderLive(live) {
    const wrap = $('live-wrap');
    if (!live || !live.length) { wrap.innerHTML = '<div class="comm-empty">Match commentary will appear here.</div>'; return; }
    let html = '';
    let lastInnings = null;
    live.forEach(entry => {
        if (entry.over !== lastInnings) {
            lastInnings = entry.over;
            html += `<div class="live-over-head">Innings ${entry.over}</div>`;
        }
        html += commLine(entry);
    });
    wrap.innerHTML = html;
}

// ---------- scorecard tab ----------
function renderScorecard(innings) {
    renderScorecardInto('sc-wrap', innings);
}

function renderScorecardInto(elementId, innings) {
    const wrap = $(elementId);
    if (!innings || !innings.length) { wrap.innerHTML = '<div class="sc-empty">No scorecard yet.</div>'; return; }
    wrap.innerHTML = innings.map(inn => {
        const batRows = inn.batting.filter(b => b.balls > 0 || b.out).map(b => {
            const sr = b.balls ? (b.runs / b.balls * 100).toFixed(1) : '0.0';
            return `<tr><td>${b.name}</td><td class="${b.out ? '' : 'not-out'}">${b.how_out}</td>
                <td>${b.runs}</td><td>${b.balls}</td><td>${b.fours}</td><td>${b.sixes}</td><td>${sr}</td></tr>`;
        }).join('');
        const bowlRows = inn.bowling.filter(b => b.balls > 0).map(b => {
            const overs = `${Math.floor(b.balls / 6)}.${b.balls % 6}`;
            const econ = b.balls ? (b.runs / (b.balls / 6)).toFixed(1) : '0.0';
            return `<tr><td>${b.name}</td><td>${overs}</td><td>${b.runs}</td><td>${b.wickets}</td><td>${econ}</td></tr>`;
        }).join('');
        return `
        <div class="sc-innings">
            <h3><span style="${inn.batting_team_color ? `color:${inn.batting_team_color}` : ''}">${inn.batting_team_name}${inn.in_progress ? ' (batting)' : ''}</span>
                <span class="tot">${inn.runs}/${inn.wickets} &nbsp; (${inn.overs})</span></h3>
            <div class="sc-sub">Batting &middot; Extras ${inn.extras}</div>
            <table>
                <thead><tr><th>Batter</th><th></th><th>R</th><th>B</th><th>4s</th><th>6s</th><th>SR</th></tr></thead>
                <tbody>${batRows || '<tr><td colspan="7">Yet to bat</td></tr>'}</tbody>
            </table>
            <div class="sc-sub">Bowling</div>
            <table>
                <thead><tr><th>Bowler</th><th>O</th><th>R</th><th>W</th><th>Econ</th></tr></thead>
                <tbody>${bowlRows || '<tr><td colspan="5">—</td></tr>'}</tbody>
            </table>
        </div>`;
    }).join('');
}

// ============================================================
//  AUCTION
// ============================================================
let aucTimerInt = null;
function setAucTimer(timeLeftMs, totalMs) {
    if (aucTimerInt) { clearInterval(aucTimerInt); aucTimerInt = null; }
    if (!totalMs) { const f = $('auc-timer-fill'); if (f) f.style.width = '0%'; return; }
    const deadline = Date.now() + timeLeftMs, total = totalMs;
    const tick = () => {
        const f = $('auc-timer-fill');
        if (!f) { clearInterval(aucTimerInt); aucTimerInt = null; return; }
        const left = Math.max(0, deadline - Date.now());
        f.style.width = (100 * left / total) + '%';
    };
    tick();
    aucTimerInt = setInterval(tick, 60);
}

// ---------- auctioneer gavel: swing animation + WebAudio "bang" ----------
let audioCtx = null;
function ensureAudio() {
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
    } catch (e) { /* audio unavailable — ignore */ }
}
function playGavel(strong) {
    ensureAudio();
    if (!audioCtx) return;
    const t = audioCtx.currentTime;
    // wooden knock: a short filtered noise burst
    const dur = 0.16, sr = audioCtx.sampleRate;
    const buf = audioCtx.createBuffer(1, Math.floor(sr * dur), sr);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 3);
    const src = audioCtx.createBufferSource(); src.buffer = buf;
    const bp = audioCtx.createBiquadFilter(); bp.type = 'bandpass';
    bp.frequency.value = strong ? 800 : 1300; bp.Q.value = 0.8;
    const g = audioCtx.createGain(); g.gain.value = strong ? 0.55 : 0.32;
    src.connect(bp); bp.connect(g); g.connect(audioCtx.destination); src.start(t);
    // low thud under it
    const o = audioCtx.createOscillator(); o.type = 'sine';
    o.frequency.setValueAtTime(200, t); o.frequency.exponentialRampToValueAtTime(60, t + 0.15);
    const og = audioCtx.createGain();
    og.gain.setValueAtTime(strong ? 0.5 : 0.28, t); og.gain.exponentialRampToValueAtTime(0.001, t + 0.2);
    o.connect(og); og.connect(audioCtx.destination); o.start(t); o.stop(t + 0.22);
}

function playVictoryFanfare() {
    ensureAudio();
    if (!audioCtx) return;
    const t0 = audioCtx.currentTime;
    const note = (freq, start, dur, vol) => {
        const o = audioCtx.createOscillator(); o.type = 'triangle';
        o.frequency.setValueAtTime(freq, start);
        const g = audioCtx.createGain();
        g.gain.setValueAtTime(0.0001, start);
        g.gain.exponentialRampToValueAtTime(vol, start + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
        o.connect(g); g.connect(audioCtx.destination);
        o.start(start); o.stop(start + dur + 0.05);
    };
    // short ascending arpeggio (C5 E5 G5 C6) then a sustained final chord
    const run = [523.25, 659.25, 783.99, 1046.50];
    run.forEach((freq, i) => note(freq, t0 + i * 0.16, 0.5, 0.32));
    const chordStart = t0 + run.length * 0.16 + 0.05;
    run.forEach(freq => note(freq, chordStart, 1.3, 0.24));
}
// ---------- ball-by-ball sound design ----------
function playDotTick() {
    ensureAudio();
    if (!audioCtx) return;
    const t = audioCtx.currentTime;
    const o = audioCtx.createOscillator(); o.type = 'square';
    o.frequency.setValueAtTime(220, t);
    const g = audioCtx.createGain();
    g.gain.setValueAtTime(0.045, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.05);
    o.connect(g); g.connect(audioCtx.destination);
    o.start(t); o.stop(t + 0.06);
}

function playBatCrack(big) {
    ensureAudio();
    if (!audioCtx) return;
    const t = audioCtx.currentTime;
    const dur = 0.06, sr = audioCtx.sampleRate;
    const buf = audioCtx.createBuffer(1, Math.floor(sr * dur), sr);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / d.length, 2);
    const src = audioCtx.createBufferSource(); src.buffer = buf;
    const bp = audioCtx.createBiquadFilter(); bp.type = 'bandpass';
    bp.frequency.value = big ? 2200 : 2800; bp.Q.value = 1.2;
    const g = audioCtx.createGain(); g.gain.value = big ? 0.6 : 0.4;
    src.connect(bp); bp.connect(g); g.connect(audioCtx.destination); src.start(t);
}

function playCrowdRoar(intensity) {
    // intensity: 1 (four) or 2 (six) -- a filtered-noise swell standing in for a crowd roar
    ensureAudio();
    if (!audioCtx) return;
    const t0 = audioCtx.currentTime;
    const dur = intensity >= 2 ? 1.5 : 0.85;
    const sr = audioCtx.sampleRate;
    const buf = audioCtx.createBuffer(1, Math.floor(sr * dur), sr);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    const src = audioCtx.createBufferSource(); src.buffer = buf;
    const bp = audioCtx.createBiquadFilter(); bp.type = 'bandpass';
    bp.frequency.setValueAtTime(500, t0);
    bp.frequency.linearRampToValueAtTime(1400, t0 + dur * 0.4);
    bp.frequency.linearRampToValueAtTime(700, t0 + dur);
    bp.Q.value = 0.6;
    const g = audioCtx.createGain();
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(intensity >= 2 ? 0.38 : 0.24, t0 + dur * 0.25);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    src.connect(bp); bp.connect(g); g.connect(audioCtx.destination);
    src.start(t0); src.stop(t0 + dur + 0.05);
}

function playWicketSting() {
    ensureAudio();
    if (!audioCtx) return;
    const t0 = audioCtx.currentTime;
    const o = audioCtx.createOscillator(); o.type = 'sawtooth';
    o.frequency.setValueAtTime(300, t0);
    o.frequency.exponentialRampToValueAtTime(85, t0 + 0.4);
    const g = audioCtx.createGain();
    g.gain.setValueAtTime(0.32, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.45);
    o.connect(g); g.connect(audioCtx.destination);
    o.start(t0); o.stop(t0 + 0.5);
    // crowd gasp
    const dur = 0.5, sr = audioCtx.sampleRate;
    const buf = audioCtx.createBuffer(1, Math.floor(sr * dur), sr);
    const d = buf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    const src = audioCtx.createBufferSource(); src.buffer = buf;
    const bp = audioCtx.createBiquadFilter(); bp.type = 'bandpass';
    bp.frequency.setValueAtTime(800, t0); bp.frequency.linearRampToValueAtTime(1600, t0 + 0.15);
    bp.Q.value = 0.7;
    const gg = audioCtx.createGain();
    gg.gain.setValueAtTime(0.0001, t0);
    gg.gain.exponentialRampToValueAtTime(0.28, t0 + 0.1);
    gg.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    src.connect(bp); bp.connect(gg); gg.connect(audioCtx.destination);
    src.start(t0); src.stop(t0 + dur + 0.05);
}

function playMilestoneSting() {
    ensureAudio();
    if (!audioCtx) return;
    const t0 = audioCtx.currentTime;
    const note = (freq, start, dur, vol) => {
        const o = audioCtx.createOscillator(); o.type = 'triangle';
        o.frequency.setValueAtTime(freq, start);
        const g = audioCtx.createGain();
        g.gain.setValueAtTime(0.0001, start);
        g.gain.exponentialRampToValueAtTime(vol, start + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, start + dur);
        o.connect(g); g.connect(audioCtx.destination);
        o.start(start); o.stop(start + dur + 0.05);
    };
    [659.25, 783.99, 1046.50].forEach((freq, i) => note(freq, t0 + i * 0.1, 0.35, 0.26));
}

function playHatTrickSting() {
    for (let i = 0; i < 3; i++) setTimeout(() => playWicketSting(), i * 180);
}

function playOutcomeSound(outcome) {
    if (outcome === '0') playDotTick();
    else if (outcome === '4') { playBatCrack(false); playCrowdRoar(1); }
    else if (outcome === '6') { playBatCrack(true); playCrowdRoar(2); }
    else if (outcome === '1' || outcome === '2' || outcome === '3' || outcome === '5') playBatCrack(false);
    else if (outcome === 'W') playWicketSting();
    else if (outcome === '50' || outcome === '100') playMilestoneSting();
    else if (outcome === 'HT') playHatTrickSting();
}

let aucFx = { strike: -1, stage: null };
function maybeGavel(a) {
    // 'preview' (before lot 1) and the very first observation after that
    // (aucFx.stage === null, e.g. right after the lobby's ready press drops
    // you onto the auction screen) only ever establish a fresh baseline --
    // never fire from them. Without this, landing on the auction screen for
    // the first time while the real stage already happened to be 'resolved'
    // (or strike already >0) read as a brand-new transition and fired the
    // gavel immediately, before any bid/sale had actually just happened.
    if (a.stage === 'preview' || aucFx.stage === null) {
        aucFx = { strike: a.stage === 'bidding' ? a.strike : 0, stage: a.stage };
        return;
    }
    const struck = a.stage === 'bidding' && a.strike > aucFx.strike && a.strike > 0;
    const resolved = a.stage === 'resolved' && aucFx.stage !== 'resolved';
    if (struck || resolved) { playGavel(resolved); swingAuctioneerHammer(); }
    aucFx = { strike: a.stage === 'bidding' ? a.strike : 0, stage: a.stage };
}

function swingAuctioneerHammer() {
    const arm = document.querySelector('.a-arm-r-pivot');
    if (arm) { arm.classList.remove('swing'); void arm.offsetWidth; arm.classList.add('swing'); }
    const podium = $('a-podium');
    if (podium) { podium.classList.remove('hit'); void podium.offsetWidth; podium.classList.add('hit'); }
}

let lastBidder = null;
function maybeFlashRivalBid(a, me) {
    if (a.stage === 'bidding' && a.active_bidder && a.active_bidder !== me && a.active_bidder !== lastBidder) {
        const holder = document.querySelector('.spot-holder');
        if (holder) {
            holder.classList.remove('rival-bid-flash');
            void holder.offsetWidth;
            holder.classList.add('rival-bid-flash');
        }
    }
    lastBidder = a.stage === 'bidding' ? a.active_bidder : null;
}

// ---------- auctioneer commentary, bucketed by real event type ----------
const AUC_LINES = {
    sold: [
        "SOLD! Please collect your player at gate number two.",
        "Gone! And just like that, someone's purse got lighter.",
        "That's a wrap on this one — moving right along.",
        "SOLD! Someone's going home very happy tonight.",
        "And down goes the hammer — that's how it's done.",
        "Sold to the team with the itchiest bidding finger.",
        "That's one squad slot filled, several more panic attacks to go.",
        "SOLD! The purse took a hit, but the smile's bigger.",
        "Deal done. Somebody's spreadsheet just got a new row.",
        "And that's a wrap — no refunds, no regrets, hopefully.",
        "SOLD! Front row seat to somebody's next big signing.",
        "Gone in a flash — didn't even see the bid coming.",
        "That's the hammer down — pack it up, next one's coming.",
        "SOLD! One happy owner, one slightly lighter wallet.",
    ],
    unsold: [
        "Unsold. Tough crowd today.",
        "No takers — even I'm surprised.",
        "Nobody wanted that one. Moving on.",
        "Unsold! Sometimes the room just isn't feeling it.",
        "Not a single hand. Brutal, honestly.",
        "That one's going back in the pool. No hard feelings.",
        "Silence. Total silence. Moving swiftly along.",
        "Unsold — even a discount wouldn't have helped there.",
        "Nobody blinked. Nobody bid. Next!",
        "That's a tumbleweed moment if I've ever seen one.",
        "Unsold! The purse strings stayed shut on this one.",
        "Zero interest. Not even a polite nibble.",
        "Well, that happened. Onward we go.",
        "Unsold — tough break, better luck in the next set.",
    ],
    marquee: [
        "Ladies and gentlemen, THIS is what you came for.",
        "Now THIS is a name that gets the room talking.",
        "Marquee lot, folks — this is where the real money moves.",
        "Everybody sit up straight, this one's a big deal.",
        "This is the moment the whole set's been building to.",
        "Marquee alert! Purses everywhere just got nervous.",
        "This is the name that ends careers. Bidding careers, that is.",
        "Big name, bigger bidding war incoming. Watch this.",
        "Marquee lot — the kind that makes owners sweat on camera.",
        "This is the headline act. Try to keep up.",
        "Now we're talking. This is the real show.",
        "Marquee set, marquee price. Buckle in.",
    ],
    overseas: [
        "Fresh off the plane and straight onto the block.",
        "A little overseas flavor for the squad, anyone?",
        "Passport stamped, bags barely unpacked, already up for bidding.",
        "Imported talent, folks — premium shipping included.",
        "This one's got frequent flyer miles AND a bat.",
        "Overseas slot up for grabs — choose wisely, only five allowed.",
        "Straight off international duty and into this auction.",
        "A visa, a kit bag, and a whole lot of expectation.",
        "This one's travelled further to get here than most of you have all year.",
        "Overseas quota's filling up fast — speak now.",
        "Jet-lagged or not, this one's ready to be bid on.",
        "Imported goods, folks. No customs duty, just crores.",
    ],
    bigjump: [
        "Whoa! Somebody really wants this one.",
        "That's a serious jump — the purse is sweating.",
        "Okay, that escalated quickly.",
        "Somebody just threw caution straight out the window.",
        "That jump just made every rival owner sit up.",
        "The gloves are off now, folks.",
        "That's not a bid, that's a statement.",
        "Purse strings just got real loose, real fast.",
        "Somebody means business all of a sudden.",
        "That jump just changed the whole mood in here.",
        "Bold. Reckless. Possibly both. Let's see where this goes.",
        "That's the sound of a strategy going out the window.",
    ],
    out: [
        "And there goes another one, folding under pressure.",
        "Smart move, or cold feet? You decide.",
        "Out! The purse lives to fight another lot.",
        "Folded like a lawn chair. Respect the restraint, I suppose.",
        "And just like that, one less competitor in the room.",
        "Out — sometimes the smartest bid is no bid at all.",
        "There's the exit. Dignity mostly intact.",
        "Pulled out — the wallet says thank you.",
        "And they're gone. Didn't even blink twice.",
        "Out! Living to bid another day, apparently.",
        "That's one fewer hand in the air. Noted.",
        "Folded quietly. No drama, just arithmetic.",
    ],
    once: [
        "Going once… my arm's getting tired, people!",
        "Going once — anyone? Anyone at all?",
        "Once! Last call before this one's someone's problem.",
        "Going once, and the room's gone very quiet.",
        "Once! Speak now, or hold your purse forever.",
        "Going once — the hammer's getting twitchy.",
        "Once! This is your moment, don't waste it.",
        "Going once, and I mean it this time. Probably.",
        "That's once. Tick tock, folks.",
        "Once! The suspense is doing wonders for my heart rate.",
        "Going once — somebody, anybody, blink.",
        "Once! This lot's about to become history.",
    ],
    twice: [
        "Going twice! Last chance, folks!",
        "Twice! Now or never, quite literally.",
        "Going twice — the hammer's loading up.",
        "Twice! This is genuinely your last shot.",
        "Going twice, and the tension is delicious.",
        "Twice! Somebody's about to be very happy or very relieved.",
        "Going twice — blink and it's gone.",
        "Twice! I can already feel the hammer coming down.",
        "Going twice, last call before the gavel falls.",
        "Twice! Speak up or forever hold your bid.",
        "Going twice — this is it, folks.",
        "Twice! The countdown's basically over.",
    ],
};
function auctioneerLine(bucket) {
    const arr = AUC_LINES[bucket] || [];
    return arr.length ? arr[Math.floor(Math.random() * arr.length)] : '';
}

let aucCommentaryState = { stage: null, strike: 0, current: null, bid: null, out: {} };
function maybeAuctioneerLine(a) {
    const bubble = $('a-bubble');
    if (!bubble) return;
    const prev = aucCommentaryState;
    let bucket = null;

    if (a.stage === 'bidding' && a.current && prev.current !== a.current.name) {
        bucket = a.current.is_foreigner ? 'overseas' : (a.tier === 'Marquee' ? 'marquee' : null);
    } else if (a.stage === 'bidding' && prev.bid !== null && a.current_bid - prev.bid >= 3) {
        bucket = 'bigjump';
    } else if (a.stage === 'bidding' && a.strike === 1 && prev.strike !== 1) {
        bucket = 'once';
    } else if (a.stage === 'bidding' && a.strike === 2 && prev.strike !== 2) {
        bucket = 'twice';
    } else if (a.stage === 'resolved' && prev.stage !== 'resolved') {
        bucket = a.last_result === 'sold' ? 'sold' : 'unsold';
    } else if (a.stage === 'bidding') {
        const newlyOut = Object.keys(a.out).find(t => a.out[t] && !prev.out[t]);
        if (newlyOut) bucket = 'out';
    }

    if (bucket) {
        const line = auctioneerLine(bucket);
        if (line) bubble.textContent = line;
    } else if (!bubble.textContent) {
        bubble.textContent = "Let's see some bids, folks!";
    }

    aucCommentaryState = {
        stage: a.stage, strike: a.stage === 'bidding' ? a.strike : 0,
        current: a.current ? a.current.name : null,
        bid: a.stage === 'bidding' ? a.current_bid : null,
        out: { ...a.out },
    };
}

// Captures scrollTop for every selector that currently matches, runs
// rebuildFn (which typically replaces innerHTML somewhere in that subtree --
// destroying and recreating those very elements), then re-applies the
// captured values to whatever now matches the same selectors. Necessary
// because .pool-list caps its own height and scrolls internally on desktop
// (its parent doesn't scroll at all in that case), while on mobile the
// .pool-list cap is lifted and an ANCESTOR scrolls instead -- so which
// element is "the" scroll container depends on viewport, and a rebuild
// always creates a brand-new .pool-list node with scrollTop reset to 0.
function withScrollPreserved(selectors, rebuildFn) {
    const prev = selectors.map(sel => {
        const el = document.querySelector(sel);
        return el ? el.scrollTop : null;
    });
    rebuildFn();
    selectors.forEach((sel, i) => {
        if (prev[i] === null) return;
        const el = document.querySelector(sel);
        if (el) el.scrollTop = prev[i];
    });
}

function renderAuction(state) {
    const a = state.auction;
    const me = a.you_role;
    $('auc-role').textContent = 'You: ' + (a.my_squad ? a.my_squad.name : me);
    $('auc-role').className = 'role-pill batting';
    const pc = $('auc-poke-counter');
    if (pc) { pc.classList.remove('hidden'); pc.innerHTML = `Pokes left: <b>${a.pokes_left}</b>`; }

    // Every poll (bid, timer tick, opponent ready...) rebuilds this whole
    // panel, which yanks any scroll position back to the top -- most
    // noticeable while scrolling My Squad's card list. The elevator ride
    // already happened in the join lobby (see renderLobby/renderTournamentLobby)
    // -- once you're in phase 'auction' you're already "in the room", so
    // every stage (including the pre-lot-1 'preview' ready-gate) renders as
    // the normal floor, never the lobby again.
    withScrollPreserved(['#auc-floor-shell', '.mine-list', '.history-list'], () => {
        const shell = $('auc-floor-shell');
        shell.innerHTML = floorShell(a, me);
        wireAuctionCenter(a, me);
        animatePurseValues(shell);
        maybeFlashRivalBid(a, me);
    });

    // Always render full pool into its own tab -- every poll rebuilds this,
    // so explicitly preserve scroll position or browsing the list while an
    // auction is live gets yanked back to the top on every single update.
    if ($('auc-full-pool')) {
        const sold = [
            ...(a.my_squad ? a.my_squad.roster.map(p => p.name) : []),
            ...(a.other_squads || []).flatMap(s => s.roster_names || []),
        ];
        withScrollPreserved(['#atab-pool', '#auc-full-pool .pool-list'], () => {
            $('auc-full-pool').innerHTML = poolList(a.pool, sold);
        });
    }

    if ($('auc-minigames')) {
        // Every poll (a bid, a strike tick, anything) was wiping out whatever
        // you'd half-typed into the guess box, since this rebuilds the whole
        // panel unconditionally -- same class of bug withScrollPreserved
        // exists for, just for an input's value/focus instead of scroll.
        const prevInput = $('guess-input');
        const prevVal = prevInput ? prevInput.value : null;
        const hadFocus = prevInput === document.activeElement;
        $('auc-minigames').innerHTML = miniGamesPanel(a, me);
        wireMiniGames();
        if (prevVal) {
            const newInput = $('guess-input');
            if (newInput) {
                newInput.value = prevVal;
                if (hadFocus) newInput.focus();
            }
        }
    }

    if (a.stage === 'bidding' || a.stage === 'done') setAucTimer(a.time_left_ms, a.total_wait_ms);
    else setAucTimer(0, 0);

    maybeGavel(a);
    maybeAuctioneerLine(a);
    maybeRollFloorBuzz(a);
    maybeShowNewEvents(a, me);
}

function initials(name) {
    return (name || '').split(' ').map(w => w[0]).filter(Boolean).slice(0, 2).join('').toUpperCase() || '?';
}

function playLobbyDoorsTransition() {
    const overlay = document.createElement('div');
    overlay.className = 'lobby-doors-transition';
    overlay.innerHTML = `<div class="lobby-doors-transition-door left"></div><div class="lobby-doors-transition-door right"></div>`;
    document.body.appendChild(overlay);
    requestAnimationFrame(() => overlay.classList.add('open'));
    setTimeout(() => overlay.remove(), 1300);
}

function floorShell(a, me) {
    return `
    <div class="auc-shell">
        <div class="auc-left">
            <div class="panel auc-mine-box">${mySquadCards(a.my_squad, a)}</div>
        </div>
        <div class="auc-center">
            ${stageBlock(a, me)}
            <div class="auc-floor" id="auc-floor">
                <div class="floor-kicker">The Floor</div>
                <div class="floor-tables" data-teams="${(a.other_squads || []).length}">${floorTables(a.other_squads, a)}</div>
            </div>
        </div>
        <div class="auc-right">${rightRail(a)}</div>
        <div class="panel auc-console-box">${consoleBox(a, me)}</div>
    </div>`;
}

function mySquadCards(sq, a) {
    const rows = sq.roster.map(p => {
        const isKeeper = p.is_keeper || p.assigned_role === 'Wicket Keeper';
        const price = typeof p.price === 'number' ? p.price.toFixed(1) : (p.price ?? 0);
        return pcard(p, {
            figure: `₹${price} Cr`,
            tag: `${p.is_foreigner ? 'OS · ' : ''}${isKeeper ? 'WK' : (p.assigned_role || '')}`,
            // home/away isn't decided until a real match -- 'home' just reads
            // as "this is your squad's colour" here, not a real jersey side
            jerseyColor: sq.color, jerseyStyle: 'home',
        });
    }).join('');
    const rosterHtml = rows
        ? `<div class="mine-list" id="mine-list">${rows}</div>`
        : '<div class="bench-empty">No buys yet.</div>';
    const pct = Math.min(100, Math.round((sq.count / a.squad_min) * 100));
    const need = Math.max(0, a.squad_min - sq.count);
    const progLabel = sq.count >= a.squad_min
        ? (sq.wk >= 1 ? 'Squad legal — can lock in' : 'Need a wicket-keeper')
        : `${need} more to reach the minimum ${a.squad_min}`;
    return `
        <div class="panel-title">My Squad — <span class="purse" data-purse="${sq.budget}" data-purse-key="my">₹${sq.budget.toFixed(1)} Cr</span> left ${sq.locked ? '<span class="badge-tier">LOCKED</span>' : ''}</div>
        <div class="squad-stats">
            <span>Squad <b>${sq.count}</b>/${a.squad_max}</span>
            <span>Overseas <b>${sq.os}</b></span>
            <span>Keepers <b>${sq.wk}</b></span>
        </div>
        <div class="squad-progress"><div class="squad-progress-fill" style="width:${pct}%"></div></div>
        <div class="squad-progress-label">${progLabel}</div>
        ${rosterHtml}`;
}

// ---------- auction theater: animated purse ticks ----------
const purseCache = {};
function animatePurseValues(root) {
    (root || document).querySelectorAll('[data-purse]').forEach(el => {
        const key = el.dataset.purseKey;
        const target = parseFloat(el.dataset.purse);
        const from = purseCache[key] !== undefined ? purseCache[key] : target;
        purseCache[key] = target;
        if (Math.abs(from - target) < 0.001) { el.textContent = `₹${target.toFixed(1)} Cr`; return; }
        el.classList.add(target < from ? 'purse-tick-down' : 'purse-tick-up');
        setTimeout(() => el.classList.remove('purse-tick-down', 'purse-tick-up'), 500);
        const start = performance.now();
        const dur = 450;
        const step = (now) => {
            const t = Math.min(1, (now - start) / dur);
            el.textContent = `₹${(from + (target - from) * t).toFixed(1)} Cr`;
            if (t < 1) requestAnimationFrame(step);
            else el.textContent = `₹${target.toFixed(1)} Cr`;
        };
        requestAnimationFrame(step);
    });
}

function teamName(a, role) {
    if (role === a.you_role) return a.my_squad.name;
    if (a.opp_squad && a.opp_squad.team_id === role) return a.opp_squad.name;
    const found = (a.other_squads || []).find(s => s.team_id === role);
    return found ? found.name : role;
}

function teamColor(a, role) {
    if (role === a.you_role) return a.my_squad.color;
    if (a.opp_squad && a.opp_squad.team_id === role) return a.opp_squad.color;
    const found = (a.other_squads || []).find(s => s.team_id === role);
    return found ? found.color : null;
}

// ---------- the floor: round tables for every other squad ----------
function floorTables(others, a) {
    if (!others || !others.length) return '<div class="bench-empty">No other squads yet.</div>';
    // No more hand-placed coordinates -- .floor-tables is a real CSS grid
    // (auto-fit/minmax), so tables get bigger with fewer opponents and wrap
    // into more rows/columns with more, instead of clustering in one corner.
    const stage = a ? a.stage : null;
    const readyMap = (a && a.ready) || {};
    const outMap = (a && a.out) || {};
    return others.map((sq) => {
        const sorted = [...sq.roster].sort((x, y) => (y.price || 0) - (x.price || 0));
        const top2 = sorted.slice(0, 2);
        const extra = sq.roster.length - top2.length;
        const cardsHtml = top2.map(p => pcard(p, {
            figure: `₹${(p.price || 0).toFixed(1)} Cr`, jerseyColor: sq.color, jerseyStyle: 'home',
        })).join('');

        // Who's holding up the next lot, and who's already folded on this one --
        // both readable straight off the table instead of only in the popup.
        // A ready-light only means anything during a ready-gate; "OUT" only
        // during live bidding.
        const inGate = stage === 'preview' || stage === 'resolved';
        const isReady = !!readyMap[sq.team_id] || sq.locked;
        const light = inGate
            ? `<span class="table-light ${isReady ? 'on' : 'off'}"
                     title="${isReady ? 'Ready' : 'Not ready yet'}"></span>`
            : '';
        const outBadge = (stage === 'bidding' && outMap[sq.team_id])
            ? '<div class="table-out-badge">OUT</div>' : '';
        // (a fold lasts only the current lot -- the server resets `out` in
        // _present_player, so the table lights back up on the next player)

        // Player names live in the squad popup, not on the cloth -- the table
        // is a glance-level "who's here / are they still in", the popup is the
        // detail view.
        const folded = stage === 'bidding' && !!outMap[sq.team_id];

        return `<div class="round-table${folded ? ' folded' : ''}" data-team="${sq.team_id}">
            <div class="table-shadow"></div>
            <div class="table-cloth">
                <div class="table-nameplate">${light}${sq.name}${sq.locked ? ' <span class="badge-tier">LOCKED</span>' : ''}</div>
                ${outBadge}
                ${extra > 0 ? `<div class="table-more">+${extra}</div>` : ''}
                <div class="table-cards">${cardsHtml || ''}</div>
            </div>
            <div class="table-stats">${sq.count} players · ₹${sq.budget.toFixed(1)} Cr left</div>
        </div>`;
    }).join('');
}

function openSquadPopup(teamId) {
    if (!CURRENT || !CURRENT.auction) return;
    const sq = (CURRENT.auction.other_squads || []).find(s => s.team_id === teamId);
    if (!sq) return;
    const rows = sq.roster.map(p => pcard(p, {
        figure: `₹${(p.price || 0).toFixed(1)} Cr`, jerseyColor: sq.color, jerseyStyle: 'home',
    })).join('');
    $('squad-popup-card').innerHTML = `
        <h2>${sq.name}</h2>
        <p class="sub">${sq.count} players · ₹${sq.budget.toFixed(1)} Cr left</p>
        <div class="modal-grid">${rows || '<div class="bench-empty">No buys yet.</div>'}</div>
        <button class="btn-ghost btn-lg" id="squad-popup-close" style="margin-top:1rem;">Close</button>`;
    $('squad-popup-overlay').classList.remove('hidden');
    $('squad-popup-close').addEventListener('click', closeSquadPopup);
}
function closeSquadPopup() { $('squad-popup-overlay').classList.add('hidden'); }

// ---------- recent buys + floor buzz ----------
function rightRail(a) {
    const rows = (a.sold_log || []).slice(0, 12).map(s =>
        `<div class="hrow"><span class="p">${s.name}</span><span class="t">${s.team}</span><span class="amt">₹${s.price.toFixed(1)}</span></div>`
    ).join('');
    return `
        <div class="panel auc-history-box">
            <div class="panel-title">Recent Buys</div>
            <div class="history-list">${rows || '<div class="bench-empty">No sales yet.</div>'}</div>
        </div>
        <div class="panel auc-news-box" id="auc-news-box">${floorBuzzHtml()}</div>`;
}

const FLOOR_BUZZ_LINES = [
    'Sources say Yuvraj Singh spent the break perfecting his "not bothered" face.',
    'A franchise owner was seen bidding by raising the wrong hand. Twice.',
    'A team’s scout was overheard calling a No. 11 "a future all-rounder, probably."',
    'Rumor mill: one owner is bidding purely to annoy a rival. It’s working.',
    'A support staffer was seen googling the player mid-auction.',
    'Someone just asked if the purse is refundable. It is not.',
    'A team’s group chat has apparently gone silent. That’s never good.',
    'Reports suggest snacks were a bigger topic than strategy this set.',
    'A franchise’s analytics team was seen arguing with its owner’s gut feeling. Gut feeling is winning.',
    'Someone bid on a player they already own. Awkward silence followed.',
    'A team’s mascot has reportedly started a betting pool on unsold lots.',
    'Word is one owner brought a lucky pen. It hasn’t worked yet.',
    'A rival scout was caught taking notes on a napkin. Bold strategy.',
    'Someone’s phone rang mid-bid. It was their mother. The bid stood.',
    'A team’s social media intern is already drafting the "we got robbed" post.',
    'Reports of a secret handshake between two rival owners. Suspicious.',
    'One franchise’s spreadsheet reportedly has more tabs than players.',
    'A support staffer was seen crossing fingers, toes, and possibly eyes.',
    'Someone just realized they’ve been bidding against their own teammate.',
    'A team’s owner asked if snacks could be traded for a discount. The answer was no.',
    'Rumor has it one franchise’s mascot costume is now for sale too.',
    'A scout’s "sure thing" pick has gone unsold twice. The napkin is under review.',
    'Someone’s laptop battery died mid-auction. Panic ensued, briefly.',
    'A team’s owner reportedly Googled "how much is a crore, actually."',
    'One franchise is rumored to be bidding by vibes alone. It’s oddly effective.',
    'A rival team’s intern was seen practicing their "we’re not worried" face.',
    'Someone asked if unsold players get a consolation prize. They do not.',
    'A team’s owner has started referring to the purse as "our life savings."',
    'Reports suggest one franchise is one bad bid away from a group therapy session.',
    'Someone’s coffee order was mistaken for a bid. Chaos, briefly.',
    'A scout was overheard saying "trust me" for the fourth time this hour.',
    'One owner’s lucky charm is reportedly a slightly bent coin.',
    'A team’s status message just changed to "it’s fine, everything’s fine."',
    'Someone bid in their sleep, allegedly. The bid still counted.',
    'A rival franchise is said to be tracking bids in a color-coded notebook.',
    'One owner keeps asking "are we the villains in this story?" No answer yet.',
    'A support staffer was seen doing math on their fingers. It checked out.',
    'Reports of a team naming their next purchase before actually buying them.',
    'Someone’s "final offer" was immediately followed by a higher one. Awkward.',
    'A franchise owner reportedly high-fived the wrong person after a win.',
    'One scout keeps a "do not buy" list that’s suspiciously long.',
    'A team’s intern was overheard asking what a wicketkeeper actually does.',
    'Someone’s bidding strategy is reportedly "just outlast everyone." Bold.',
    'A rival owner was seen consulting a fortune cookie before bidding.',
    'One franchise’s group chat just sent 47 messages in a row. All caps.',
    'A scout’s confidence dropped the instant the bidding actually started.',
    'Someone asked if they could pay in installments. They could not.',
    'A team’s owner is reportedly keeping a "regrets" list for later.',
    'One franchise has apparently nicknamed a rival "The Wallet Slayer."',
    'A support staffer’s "just one more bid" has been said six times already.',
    'Reports suggest a team’s mascot has stronger opinions than the coach.',
    'Someone’s "I’m out" was said with visible relief. Very visible.',
    'A rival scout was seen double-checking a player’s stats. Twice. Same stats.',
    'One owner’s strategy is reportedly "bid high, ask questions later."',
    'A team’s spreadsheet formula broke. Somehow the purse still balanced.',
    'Someone just found out you can’t un-bid. Learning experience.',
    'A franchise owner was overheard practicing their post-win speech early.',
    'One scout’s "sleeper pick" has now been called out three sets running.',
    'A team’s intern accidentally liked a rival’s social media post. Tension.',
    'Reports of an owner negotiating with themselves out loud. It got heated.',
];
let floorBuzzShown = [];
function floorBuzzHtml() {
    if (!floorBuzzShown.length) rollFloorBuzz();
    return `<div class="panel-title">Floor Buzz</div>` +
        floorBuzzShown.map(l => `<div class="news-item">${l}</div>`).join('');
}
function rollFloorBuzz() {
    const pool = FLOOR_BUZZ_LINES.filter(l => !floorBuzzShown.includes(l));
    const src = pool.length ? pool : FLOOR_BUZZ_LINES;
    floorBuzzShown = [src[Math.floor(Math.random() * src.length)], ...floorBuzzShown].slice(0, 3);
}
let lastBuzzSoldCount = null;
function maybeRollFloorBuzz(a) {
    const n = (a.sold_log || []).length;
    // deliberately NOT tied to every lot -- only a ~20% chance per sale, so
    // it reads as background gossip rather than a per-event commentary track
    if (lastBuzzSoldCount !== null && n > lastBuzzSoldCount && Math.random() < 0.2) {
        rollFloorBuzz();
        const box = $('auc-news-box');
        if (box) box.innerHTML = floorBuzzHtml();
    }
    lastBuzzSoldCount = n;
}

// ---------- poke / banter: a floating menu that escapes .auc-floor's
// overflow:hidden by living outside it, positioned via getBoundingClientRect
// and clamped to the viewport so it can never be clipped, however close the
// hovered table is to the top/right/bottom edge of the floor. ----------
const BANTER_LINES = [
    'Purse looking thin?', "That's a robbery.", 'Bold. Very bold.',
    'Overpaid, legend.', 'Marquee price, bench player?', 'Yikes. Just yikes.',
    'Can I borrow 5cr?',
];
let interactHideTimer = null;
function showTableInteractFor(tableEl) {
    clearTimeout(interactHideTimer);
    const team = tableEl.dataset.team;
    const pokesLeft = (CURRENT && CURRENT.auction) ? CURRENT.auction.pokes_left : 0;
    const menu = $('auc-table-interact');
    menu.dataset.team = team;
    menu.innerHTML = `
        <button class="interact-btn poke" ${pokesLeft <= 0 ? 'disabled' : ''} data-poke="${team}">Poke</button>
        <button class="interact-btn" data-banter-toggle="${team}">Banter</button>
        <div class="banter-pop" id="banter-pop"></div>`;
    positionInteractMenu(tableEl, menu);
    menu.classList.remove('hidden');
}
function positionInteractMenu(tableEl, menu) {
    const rect = tableEl.getBoundingClientRect();
    const w = 190;
    let left = rect.left + rect.width / 2 - w / 2;
    left = Math.max(8, Math.min(window.innerWidth - w - 8, left));
    let top = rect.top - 44;
    if (top < 8) top = rect.bottom + 8;   // flip below when too close to the top edge
    menu.style.left = left + 'px';
    menu.style.top = top + 'px';
}
function scheduleHideInteract() {
    clearTimeout(interactHideTimer);
    interactHideTimer = setTimeout(() => $('auc-table-interact').classList.add('hidden'), 300);
}

function spawnCrowdReaction(anchorEl) {
    const react = document.createElement('div');
    react.className = 'poke-reaction';
    react.innerHTML = '<span>HA</span><span>HA</span><span>HA</span>';
    const crowd = document.createElement('div');
    crowd.className = 'poke-crowd';
    crowd.innerHTML = '<div class="face"></div><div class="face"></div><div class="face"></div>';
    anchorEl.appendChild(react);
    anchorEl.appendChild(crowd);
    setTimeout(() => { react.remove(); crowd.remove(); }, 1600);
}

// Poke/banter notifications are playful, not errors -- toast() is the site's
// red "something went wrong" banner (see its .toast CSS: var(--leather),
// bottom-center). Reusing it for these made a poke read as a warning.
function floorToast(msg) {
    const t = document.createElement('div');
    t.className = 'floor-toast';
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3200);
}

let seenAucEventIds = new Set();
function maybeShowNewEvents(a, me) {
    const now = Date.now() / 1000;
    (a.events || []).forEach(ev => {
        if (seenAucEventIds.has(ev.id)) return;
        seenAucEventIds.add(ev.id);
        // #auc-floor-shell gets torn down and rebuilt on every poll regardless
        // of which inner tab is showing, so switching to Mini Games/Available
        // Players and back was replaying the fall animation for an event that
        // actually happened while you were away -- gate the dramatic visuals
        // (not the toast, that's still useful late) to genuinely fresh events,
        // and explicitly clear the class afterward so it can never linger.
        const isFresh = (now - ev.ts) < 6;
        if (ev.kind === 'poke') {
            if (ev.to === me) {
                // being poked: your own bench cards fall and shake, plus a
                // quiet toast -- no HA-HA/crowd here, that mockery plays out
                // on the floor (everyone else's view of you), not your own
                floorToast(`${teamName(a, ev.from)} poked you!`);
                if (isFresh) {
                    const mine = document.getElementById('mine-list');
                    if (mine) {
                        mine.classList.remove('poked'); void mine.offsetWidth; mine.classList.add('poked');
                        setTimeout(() => mine.classList.remove('poked'), 1700);
                    }
                }
            } else if (isFresh) {
                const tableEl = document.querySelector(`.round-table[data-team="${ev.to}"]`);
                if (tableEl) {
                    const cards = tableEl.querySelector('.table-cards');
                    if (cards) {
                        cards.classList.remove('poked'); void cards.offsetWidth; cards.classList.add('poked');
                        setTimeout(() => cards.classList.remove('poked'), 1800);
                    }
                    spawnCrowdReaction(tableEl);
                }
            }
        } else if (ev.kind === 'banter') {
            if (ev.to === me) floorToast(`${teamName(a, ev.from)}: "${ev.line}"`);
            else if (ev.from === me) floorToast(`You told ${teamName(a, ev.to)}: "${ev.line}"`);
        }
    });
}

// ---------- Guess the Price (Mini Games tab) ----------
function miniGamesPanel(a) {
    const lb = (a.guess.leaderboard || []).map(row =>
        `<tr><td>${row.name}</td><td>${row.points}</td></tr>`).join('');
    const canGuess = a.stage === 'bidding' && !a.guess.locked;
    const lockedHtml = a.guess.locked
        ? `<div class="guess-locked">Your guess is ₹${a.guess.my_guess.toFixed(1)} Cr. Locked in, waiting for the hammer.</div>`
        : '';
    const lotLine = a.stage === 'bidding' && a.current
        ? `Current player: ${a.current.name}.`
        : 'No player is up for auction right now. Check back once the next one starts.';
    return `
    <div class="mg-card">
        <div class="mg-title">Guess the Price</div>
        <div class="mg-rules">Guess what the current player will sell for. You can only place a guess while that player's auction is live, one guess per player, locked in before it sells. Once it sells, the three closest guesses earn points: five for closest, three for second, one for third. Points add up across the whole auction. Purely for fun, it never touches your purse or your bids.</div>
        <div class="mg-sub">${lotLine}</div>
        ${canGuess ? `
        <div class="guess-input-row">
            <input type="number" id="guess-input" placeholder="₹ Cr" step="0.1" min="0">
            <button id="guess-lock-btn">Lock In</button>
        </div>` : ''}
        ${lockedHtml}
        <div class="panel-title" style="margin-top:1.1rem;">Leaderboard, running total</div>
        <table class="lb-table"><tr><th>Team</th><th>Points</th></tr>${lb || '<tr><td colspan="2">No guesses yet.</td></tr>'}</table>
    </div>`;
}
function wireMiniGames() {
    const btn = $('guess-lock-btn');
    if (btn) btn.addEventListener('click', () => {
        const v = parseFloat($('guess-input').value);
        if (isNaN(v) || v <= 0) return;
        aucAction('/api/guess_price', { amount: v });
    });
}


function poolList(pool, soldNames = []) {
    if (!pool) return '';
    const bySet = {};
    pool.forEach(p => { (bySet[p.set_id] = bySet[p.set_id] || { tier: p.tier, role: p.role, players: [] }).players.push(p); });
    const blocks = Object.keys(bySet).map(id => {
        const s = bySet[id];
        const items = s.players.map(p => {
            const isSold = soldNames.includes(p.name);
            return `<div class="pool-item ${isSold ? 'sold' : ''}" style="${isSold ? 'opacity:0.4; text-decoration:line-through; font-style:italic;' : ''}">${p.name} ${p.is_foreigner ? '<span class="os">OS</span>' : ''} ${p.is_keeper ? '<span class="os">WK</span>' : ''}
             <span class="ovrs">${p.batting_ovr}/${p.bowling_ovr}</span></div>`;
        }).join('');
        return `<div class="pool-set"><h5>Set ${id}: ${s.tier} ${s.role}s</h5>${items}</div>`;
    }).join('');
    return `<div class="pool-list">${blocks}</div>`;
}

function readyBlock(a, me, label) {
    let statusHtml;
    if (a.opp_squad) {
        const oppReady = a.ready[a.opp_squad.team_id] || a.opp_locked;
        statusHtml = `Opponent: ${oppReady ? 'ready' : 'not ready yet'}`;
    } else {
        const others = a.other_squads || [];
        const readyCount = others.filter(s => a.ready[s.team_id] || s.locked).length;
        statusHtml = `Other teams ready: ${readyCount}/${others.length}`;
    }
    return `<button class="btn-go btn-lg" id="auc-ready" ${a.ready[me] ? 'disabled' : ''}>
                ${a.ready[me] ? 'Ready' : label}</button>
            <div class="auc-holder">${statusHtml}</div>`;
}

function lockBlock(a) {
    if (a.my_locked) return `<div class="auc-holder">Your squad is LOCKED — waiting for the others.</div>`;
    if (a.can_lock) return `<button class="btn-gold btn-lg" id="auc-lock">Lock In Squad (${a.my_squad.count} players)</button>`;
    return '';
}

function skipSetBlock(a) {
    if (a.my_locked || !a.skip_set) return '';
    const sv = a.skip_set;
    if (sv.blocked) return `<div class="auc-holder" title="Not enough players would be left for everyone to build a full squad.">Can't skip this set — too few players would be left</div>`;
    if (sv.i_voted) return `<div class="auc-holder">Voted to skip this set (${sv.count}/${sv.total})</div>`;
    return `<button class="btn-ghost" id="auc-skip-set">Vote to skip this set (${sv.count}/${sv.total})</button>`;
}

// ---------- stage: the auctioneer + the spotlighted current/just-resolved lot ----------
function stageBlock(a, me) {
    return `<div class="auc-stage">${auctioneerHtml()}${spotlightHtml(a, me)}</div>`;
}

function auctioneerHtml() {
    return `<div class="auctioneer-wrap">
        <div class="a-bubble" id="a-bubble"></div>
        <div class="auctioneer">
            <div class="a-turban"><div class="a-turban-jewel"></div></div>
            <div class="a-head"></div>
            <div class="a-eye l"></div><div class="a-eye r"></div>
            <div class="a-mustache"></div>
            <div class="a-body"></div>
            <div class="a-sash"></div>
            <div class="a-arm-l"></div>
            <div class="a-arm-r-pivot"><div class="a-arm-r"><div class="a-hammer"></div></div></div>
        </div>
        <div class="a-podium" id="a-podium"><div class="a-impact" id="a-impact"></div></div>
    </div>`;
}

function spotlightHtml(a, me) {
    if (a.stage === 'preview') {
        return `<div class="spotlight"><div class="spot-kicker">Welcome To The Floor</div><div class="spot-msg">${a.message}</div></div>`;
    }
    if (a.stage === 'bidding' && a.current) {
        const c = a.current;
        let holder = 'No bids yet — opening price.';
        if (a.active_bidder) {
            const bc = teamColor(a, a.active_bidder);
            holder = a.active_bidder === me ? 'You hold the top bid'
                : `<span style="${bc ? `color:${bc}` : ''}">${teamName(a, a.active_bidder)}</span> holds the bid`;
        }
        const strikeTxt = a.strike === 1 ? 'Going once…' : a.strike === 2 ? 'Going twice!' : '';
        return `<div class="spotlight">
            <div class="spot-kicker">On The Block</div>
            <div class="spot-tags">
                <span class="spot-tag">${a.tier} Set</span>
                <span class="spot-tag">${a.role_name}</span>
                ${c.is_foreigner ? '<span class="spot-tag os">Overseas</span>' : ''}
            </div>
            <div class="spot-card">${pcard(c, {})}</div>
            <div class="auc-timer"><div class="auc-timer-fill" id="auc-timer-fill"></div></div>
            <div class="spot-bid-row"><div class="spot-bid">₹${a.current_bid.toFixed(1)} Cr</div><div class="spot-holder">${holder}</div></div>
            <div class="spot-strike">${strikeTxt}</div>
        </div>`;
    }
    if (a.stage === 'resolved' && a.current) {
        const c = a.current;
        const sold = a.last_result === 'sold';
        return `<div class="spotlight resolved-spot">
            <div class="spot-stamp ${sold ? 'stamp-sold' : 'stamp-unsold'}">${sold ? 'SOLD' : 'UNSOLD'}</div>
            <div class="spot-kicker">${a.tier} Set · ${a.role_name}</div>
            <div class="spot-card">${pcard(c, {})}</div>
            <div class="spot-msg" style="color:${sold ? 'var(--green-go)' : 'var(--leather)'}">${a.message}</div>
        </div>`;
    }
    // done
    return `<div class="spotlight"><div class="spot-kicker">Auction Complete</div><div class="spot-msg">${a.message}</div></div>`;
}

// ---------- console: bid controls (mechanics unchanged), lock/skip-set, Eat Snacks ----------
function consoleBox(a, me) {
    const eatSnacks = `
        <div class="snack-row ${a.auto_ready ? 'on' : ''}" id="auc-snack-toggle">
            <span class="snack-label">Eat Snacks</span>
            <div class="switch"></div>
        </div>
        <div class="snack-hint">
            <span class="off-txt">Toggle on to auto-ready between lots — sit back and watch.</span>
            <span class="on-txt">Relaxing — auto-readying for you between lots.</span>
        </div>`;

    if (a.stage === 'preview') {
        return `<div class="panel-title">Auction Console</div>
            ${readyBlock(a, me, "I'm Ready to Bid")}
            ${eatSnacks}`;
    }

    if (a.stage === 'bidding') {
        // Leading bidders can't fold -- otherwise, once every rival has
        // pulled out, the leader could pull out too and force the lot
        // unsold for free instead of paying the bid they'd otherwise win.
        const iAmLeading = a.active_bidder === me;
        // Also fixed-position: a leader can't fold, but the button keeps its
        // slot (disabled) rather than vanishing and dragging the row around.
        const pullOutBtn = `<button class="btn-red" id="bid-out" ${iAmLeading ? 'disabled' : ''}
            title="${iAmLeading ? "You hold the top bid — you can't pull out" : 'Fold on this lot'}">I'm Out</button>`;
        // Marquee lots open big (₹2 Cr) and usually move in big jumps early on --
        // +0.1 only clutters that. It shows up once the bid actually gets
        // expensive (>₹10 Cr), where fine control starts to matter. Every
        // other tier keeps +0.1 available from the start.
        const showTenth = a.tier !== 'Marquee' || a.current_bid > 10;
        // No free-text custom amount, and no plain "Bid" once someone's ahead --
        // the ONLY ways to move the price are: claim the untaken opening price
        // (nobody's bid yet), or a fixed "+" raise (matches the server, which
        // rejects a zero-amount bid once active_bidder is set).
        //
        // EVERY control keeps a FIXED position for the whole lot. The claim
        // button used to be swapped OUT for the +N row the instant someone
        // bid, which shifted every button under a finger already travelling
        // to the screen: two people going for the opening bid together meant
        // the loser's tap landed on whatever slid into that spot -- +5 or
        // +10, an accidental multi-crore raise. Now the claim button stays
        // put and just goes dead, and the +N row is always rendered (greyed
        // until it's actually legal to raise).
        const claimTaken = !!a.active_bidder;
        const claimBtn = `<button class="btn-go bid-claim-btn${claimTaken ? ' spent' : ''}"
                 id="bid-claim" ${claimTaken ? 'disabled' : ''}>
                 ${claimTaken ? `Opening bid taken — ₹${a.current_bid.toFixed(1)} Cr` : `Bid ₹${a.current_bid.toFixed(1)} Cr`}
               </button>`;
        const raise = (amt, label) =>
            `<button data-bid="${amt}" ${claimTaken ? '' : 'disabled'}>${label}</button>`;
        // +0.1 keeps its slot reserved even when hidden, so nothing else moves
        const bidControls = `${claimBtn}
               <div class="quick-adds">
                 ${showTenth ? raise('0.1', '+0.1') : '<button class="slot-hidden" disabled>+0.1</button>'}
                 ${raise('0.2', '+0.2')}${raise('0.5', '+0.5')}${raise('1', '+1')}
                 ${raise('2', '+2')}${raise('5', '+5')}${raise('10', '+10')}
               </div>`;
        // Sitting this lot out still renders the SAME skeleton (everything
        // disabled) rather than collapsing to a one-line message, so the
        // console never changes shape mid-lot.
        const sittingOut = a.out[me] || a.my_locked;
        const controls = sittingOut
            ? `<div class="bid-controls is-out">
                 <div class="auc-holder">${a.my_locked ? 'Your squad is locked — sitting out.' : 'You pulled out of this lot.'}</div>
                 ${bidControls.replace(/<button /g, '<button disabled ')}
                 <div class="bid-row"><button class="btn-red" disabled>I'm Out</button></div>
               </div>`
            : `<div class="bid-controls">${bidControls}<div class="bid-row">${pullOutBtn}</div></div>`;
        return `<div class="panel-title">Auction Console</div>
            ${controls}
            ${lockBlock(a)}
            ${skipSetBlock(a)}
            ${eatSnacks}`;
    }

    if (a.stage === 'resolved') {
        return `<div class="panel-title">Auction Console</div>
            ${readyBlock(a, me, 'Ready for Next Lot')}
            ${lockBlock(a)}
            ${skipSetBlock(a)}
            ${eatSnacks}`;
    }

    // done
    const sq = a.my_squad;
    let hint = '';
    if (sq.count < a.squad_min) hint = `Only ${sq.count}/${a.squad_min} players — you'll be kicked out when the timer runs out.`;
    else if (sq.wk < 1) hint = 'Need at least 1 wicket-keeper.';
    const secsLeft = Math.max(0, Math.ceil((a.time_left_ms || 0) / 1000));
    const forfeitWarning = a.opp_squad ? 'or you auto-forfeit!' : 'or you are kicked out!';
    const countdown = !a.my_locked
        ? `<div class="auc-holder" style="color:${secsLeft <= 15 ? 'var(--leather)' : 'var(--gold)'}">
             ${secsLeft}s to lock a valid squad ${forfeitWarning}</div>
           <div class="auc-timer"><div class="auc-timer-fill" id="auc-timer-fill"></div></div>`
        : '';
    const othersStatus = a.opp_squad
        ? `Opponent: ${a.opp_locked ? 'locked' : 'still building…'}`
        : `Others locked: ${(a.other_squads || []).filter(s => s.locked).length}/${(a.other_squads || []).length}`;
    return `<div class="panel-title">Auction Console</div>
        ${countdown}
        <div class="auc-holder">Your squad: ${sq.count}/${a.squad_max} · keepers ${sq.wk} · overseas ${sq.os}</div>
        ${lockBlock(a)}
        <div class="auc-holder">${hint}</div>
        <div class="auc-holder">${othersStatus}</div>`;
}

function wireAuctionCenter(a, me) {
    const ready = $('auc-ready'); if (ready) ready.addEventListener('click', () => aucAction('/api/auction_ready'));
    document.querySelectorAll('#auc-floor-shell [data-bid]').forEach(b =>
        b.addEventListener('click', () => aucAction('/api/bid', { amount: parseFloat(b.dataset.bid) })));
    // claims the lot's untaken opening price -- only shown while nobody's bid yet
    const claim = $('bid-claim');
    if (claim) claim.addEventListener('click', () => aucAction('/api/bid', { amount: 0 }));
    const out = $('bid-out'); if (out) out.addEventListener('click', () => aucAction('/api/pull_out'));
    const lk = $('auc-lock'); if (lk) lk.addEventListener('click', () => aucAction('/api/lock_squad'));
    const skip = $('auc-skip-set'); if (skip) skip.addEventListener('click', () => aucAction('/api/vote_skip_set'));
    const snack = $('auc-snack-toggle'); if (snack) snack.addEventListener('click', () => aucAction('/api/toggle_auto_ready'));
}

async function aucAction(path, body = {}) {
    ensureAudio();   // unlock WebAudio on the first user gesture
    try { await Net.post(path, { token: Net.getToken(), ...body }); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
}

// ============================================================
//  XI SELECTION
// ============================================================
// ---------- home ground selection (phase == 'grounds') ----------
function renderGrounds(state) {
    const gr = state.grounds;
    $('grounds-role').textContent = 'You: ' + (state.you.name || '');
    $('grounds-role').className = 'role-pill batting';
    const cards = gr.stadiums.map(s => {
        const takenByOther = s.claimed_by && !s.claimed_by_me;
        const cls = ['ground-card', s.pitch];
        if (s.claimed_by_me) cls.push('mine');
        if (takenByOther) cls.push('taken');
        return `<div class="${cls.join(' ')}" ${(!takenByOther && !gr.my_locked) ? `data-ground="${s.id}"` : ''}>
            <div class="ground-name">${s.name}</div>
            <div class="ground-city">${s.city}</div>
            <div class="pitch-chip ${s.pitch}">${s.pitch} pitch</div>
            <div class="ground-desc">${s.pitch_desc}</div>
            ${s.claimed_by ? `<div class="ground-claim">${s.claimed_by_me ? 'Your home ground' : `Claimed by ${s.claimed_by}`}</div>` : ''}
        </div>`;
    }).join('');
    $('grounds-main').innerHTML = `
        <div class="grounds-hint">Every team claims a different home ground — its pitch shapes your matches.
            Round-robin fixtures are hosted fairly, rotating between each pair's two grounds over the tournament.
            Coordinate below, then lock in. ${gr.locked_count}/${gr.total_teams} locked.</div>
        <div class="grounds-grid">${cards}</div>
        <button class="btn-go btn-lg" id="btn-lock-ground" ${(gr.my_ground && !gr.my_locked) ? '' : 'disabled'}>
            ${gr.my_locked ? 'Locked — waiting for the others…' : 'Lock In Home Ground'}</button>`;
    if (!gr.my_locked) {
        document.querySelectorAll('#grounds-main [data-ground]').forEach(el =>
            el.addEventListener('click', () => aucAction('/api/claim_ground', { ground_id: el.dataset.ground })));
    }
    const lk = $('btn-lock-ground');
    if (lk) lk.addEventListener('click', () => aucAction('/api/lock_ground'));
}

function renderXI(state) {
    // Same screen serves two callers: the plain 1v1 game's one-time XI pick
    // (state.xi) and a tournament's per-fixture reselect (state.fixture_xi,
    // used for EVERY fixture including the first) -- only the data source
    // and API endpoints differ.
    const isFixture = !!state.fixture_xi;
    const x = isFixture ? state.fixture_xi : state.xi;
    const eps = isFixture
        ? { toggle: '/api/toggle_fixture_xi', lock: '/api/lock_fixture_xi', reorder: '/api/reorder_fixture_xi' }
        : { toggle: '/api/toggle_xi', lock: '/api/lock_xi', reorder: '/api/reorder_xi' };
    $('xi-role').textContent = 'You: ' + x.team_name;
    $('xi-role').className = 'role-pill batting';
    if ($('xi-title')) {
        $('xi-title').textContent = isFixture ? `Select Your XI vs ${x.opponent_name}` : 'Select Your Playing XI';
    }

    const osBad = x.os > x.max_os, wkBad = x.wk < 1, cntOk = x.count === x.size;
    const bench = x.roster.filter(p => !p.selected);
    // chosen list follows x.xi's own order (not roster order), so a
    // drag-reorder sticks across the next poll's re-render
    const byName = {}; x.roster.forEach(p => { byName[p.name] = p; });
    const chosen = x.xi.map(n => byName[n]).filter(Boolean);
    const othersStatus = x.opponent_locked !== null
        ? `Opponent: ${x.opponent_locked ? 'locked' : 'selecting…'}`
        : `Others locked: ${x.others_locked_count}/${x.others_total}`;

    const groundBanner = x.ground
        ? `<div class="xi-ground-banner"><b>${x.ground.name}</b>, ${x.ground.city} &middot;
             <span class="pitch-chip ${x.ground.pitch}">${x.ground.pitch} pitch</span> — ${x.ground.pitch_desc}</div>`
        : '';
    $('xi-main').innerHTML = `
        ${groundBanner}
        <div class="xi-stats">
            <span class="${cntOk ? 'good' : ''}">Selected ${x.count}/${x.size}</span>
            <span class="${osBad ? 'bad' : ''}">Overseas ${x.os}/${x.max_os}</span>
            <span class="${wkBad ? 'bad' : 'good'}">Keepers ${x.wk}</span>
            ${x.locked ? '<span class="good">LOCKED</span>' : ''}
            <span style="margin-left:auto">${othersStatus}</span>
        </div>
        <div class="xi-cols">
            <div class="xi-col"><h4>Squad Bench (${bench.length})</h4><div class="xi-list" id="xi-bench"></div></div>
            <div class="xi-col"><h4>Playing XI (${chosen.length}/${x.size})
                ${!x.locked && chosen.length > 1 ? '<span class="xi-reorder-hint">drag to reorder</span>' : ''}</h4>
                <div class="xi-list" id="xi-chosen"></div></div>
        </div>
        <button class="btn-go btn-lg" id="xi-lock" ${(cntOk && !osBad && !wkBad && !x.locked) ? '' : 'disabled'}>
            ${x.locked ? 'XI Locked — waiting for the others…' : 'Lock In XI'}</button>`;

    $('xi-bench').innerHTML = bench.map(p => xiCard(p, x, false)).join('') || '<div class="bench-empty">—</div>';
    $('xi-chosen').innerHTML = chosen.map(p => xiCard(p, x, !x.locked)).join('') || '<div class="bench-empty">Tap players to add them</div>';

    if (!x.locked) {
        document.querySelectorAll('#xi-main [data-xi]').forEach(el =>
            el.addEventListener('click', () => aucAction(eps.toggle, { player_name: el.dataset.xi })));
        wireXiReorder(chosen.map(p => p.name), eps.reorder);
    }
    const lk = $('xi-lock'); if (lk) lk.addEventListener('click', () => aucAction(eps.lock));
}

function xiCard(p, x, draggable) {
    const isKeeper = p.is_keeper || p.assigned_role === 'Wicket Keeper';
    const dragAttrs = draggable ? ` draggable="true" data-xi-drag="${p.name}"` : '';
    // tournament energy meter: rest-planning gauge + the real (small) stat hit
    let energyHtml = '';
    if (p.energy !== undefined && p.energy < 100) {
        const level = p.energy >= 80 ? 'ok' : p.energy >= 50 ? 'tired' : 'gassed';
        energyHtml = `<div class="energy-bar ${level}" title="Stats -${p.fatigue_penalty}% until rested">
            <div class="energy-fill" style="width:${p.energy}%"></div></div>`;
    }
    return pcard(p, {
        ovr: Math.max(p.batting_ovr || 0, p.bowling_ovr || 0),
        selectable: !x.locked, selected: p.selected,
        attrs: `${!x.locked ? `data-xi="${p.name}"` : ''}${dragAttrs}`,
        tag: `${p.is_foreigner ? 'OS · ' : ''}${isKeeper ? 'WK' : (p.assigned_role || '')}`,
        extraHtml: energyHtml,
        jerseyColor: x.team_color, jerseyStyle: x.team_jersey,
    });
}

function wireXiReorder(order, reorderEndpoint) {
    const container = $('xi-chosen');
    if (!container) return;
    let dragName = null;
    container.querySelectorAll('[data-xi-drag]').forEach(el => {
        el.addEventListener('dragstart', () => { dragName = el.dataset.xiDrag; el.classList.add('dragging'); });
        el.addEventListener('dragend', () => el.classList.remove('dragging'));
        el.addEventListener('dragover', (e) => e.preventDefault());
        el.addEventListener('drop', (e) => {
            e.preventDefault();
            const targetName = el.dataset.xiDrag;
            if (!dragName || dragName === targetName) return;
            const next = order.slice();
            const from = next.indexOf(dragName);
            const to = next.indexOf(targetName);
            if (from === -1 || to === -1) return;
            next.splice(from, 1);
            next.splice(to, 0, dragName);
            aucAction(reorderEndpoint, { order: next });
        });
    });
}

// ---------- boot ----------
if (Net.getToken()) Net.startPolling();
