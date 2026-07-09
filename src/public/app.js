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
    selectedBowler: null,
    openerPicks: [],
};

function getBatterIntent(name) { return name in ui.batterIntents ? ui.batterIntents[name] : 50; }
function setBatterIntent(name, v) { ui.batterIntents[name] = v; }

// tracks the ball-by-ball reveal of the current over
const overAnim = { key: null, shown: 0 };

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
$('btn-create').addEventListener('click', async () => {
    try {
        const data = await Net.post('/api/create_game', { name: $('create-name').value.trim() });
        Net.setToken(data.token);
        Net.startPolling();
    } catch (e) { $('landing-err').textContent = e.message; }
});

$('btn-join').addEventListener('click', async () => {
    const code = $('join-code').value.trim().toUpperCase();
    if (code.length !== 4) { $('landing-err').textContent = 'Enter the 4-character code.'; return; }
    try {
        const data = await Net.post('/api/join_game', { code, name: $('join-name').value.trim() });
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
        });
        Net.setToken(data.token);
        Net.startPolling();
    } catch (e) { $('t-landing-err').textContent = e.message; }
});

$('btn-join-tournament').addEventListener('click', async () => {
    const code = $('t-join-code').value.trim().toUpperCase();
    if (code.length !== 4) { $('t-landing-err').textContent = 'Enter the 4-character code.'; return; }
    try {
        const data = await Net.post('/api/join_tournament', { code, name: $('t-join-name').value.trim() });
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

$('btn-new-game').addEventListener('click', () => { localStorage.removeItem('ca_token'); location.reload(); });

$('btn-back-to-tournament').addEventListener('click', () => {
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

// ---------- master render ----------
window.addEventListener('gamestate', (e) => {
    CURRENT = e.detail;
    render(CURRENT);
});

function render(state) {
    if (!state || state.status === 'no_game' || !state.you || !state.you.joined) {
        $('exit-btn').classList.add('hidden');
        $('next-fixture-overlay').classList.add('hidden');
        showScreen('landing');
        return;
    }
    // opponent (or you) left -> show result, offer back to main
    if (state.abandoned) {
        $('exit-btn').classList.add('hidden');
        $('next-fixture-overlay').classList.add('hidden');
        $('result-text').textContent = state.you_won
            ? '🏆 You win! Your opponent left the match.'
            : (state.ended_result || 'Match abandoned.');
        $('result-banner').classList.remove('hidden');
        return;
    }
    // exit/leave button: shown during play and while browsing the final
    // scorecard; hidden only while the result banner itself is up.
    const finished = state.phase === 'finished';
    $('exit-btn').classList.toggle('hidden', finished && !viewingScorecard);

    // the "your match is next, ready?" overlay floats on top of whatever
    // screen is behind it (match, bracket, ...) — independent of routing
    renderNextFixtureOverlay(state);

    // manualBracketView only makes sense while stuck looking at our own
    // finished-fixture banner with nothing queued for us yet — clear it the
    // moment that stops being true so future banners behave normally again.
    if (!(finished && state.is_tournament && !state.next_fixture)) {
        manualBracketView = false;
    }

    if (state.phase === 'lobby') {
        if (state.is_tournament) { showScreen('t-lobby'); renderTournamentLobby(state); }
        else { showScreen('lobby'); renderLobby(state); }
        return;
    }
    if (state.phase === 'auction') { showScreen('auction'); renderAuction(state); return; }
    if (state.phase === 'xi') { showScreen('xi-screen'); renderXI(state); return; }

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
    // tournament: just played, nothing queued for us yet, and we chose to
    // stop looking at the old result banner -> show the bracket instead
    if (finished && state.is_tournament && manualBracketView) {
        showScreen('t-bracket');
        renderBracket(state);
        return;
    }

    showScreen('game');
    renderGame(state);

    if (finished && !viewingScorecard) {
        $('result-text').textContent = (state.match && state.match.result) || 'Match complete';
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

// ---------- next-fixture ready overlay (tournament) ----------
function renderNextFixtureOverlay(state) {
    const nf = state.next_fixture;
    if (!nf) { $('next-fixture-overlay').classList.add('hidden'); return; }
    const card = $('next-fixture-card');
    if (nf.i_ready) {
        card.innerHTML = `
            <h2>Ready ✔</h2>
            <div class="sub">${nf.a_name} <b>vs</b> ${nf.b_name} — ${nf.kind.replace('_', ' ')}</div>
            <div class="wait-note" style="margin-top:0.8rem;">
                Waiting for ${nf.opponent_ready ? 'the match to start…' : 'the other team to ready up…'}</div>`;
    } else {
        card.innerHTML = `
            <h2>🏏 Your Match Is Next!</h2>
            <div class="sub">${nf.a_name} <b>vs</b> ${nf.b_name} — ${nf.kind.replace('_', ' ')}</div>
            <button class="btn-go btn-lg" id="btn-fixture-ready" style="margin-top:1rem;">I'm Ready ✔</button>`;
        const b = document.getElementById('btn-fixture-ready');
        if (b) b.addEventListener('click', async () => {
            try { await Net.post('/api/tournament_ready', { token: Net.getToken() }); Net.forceRefresh(); }
            catch (e) { toast(e.message); }
        });
    }
    $('next-fixture-overlay').classList.remove('hidden');
}

// ---------- lobby ----------
function renderLobby(state) {
    $('lobby-code').textContent = state.code;
    $('lobby-t1').textContent = state.teams.team1.name;
    $('lobby-t2').textContent = state.teams.team2.joined ? state.teams.team2.name : 'Waiting…';
    $('dot-t1').classList.toggle('on', state.teams.team1.joined);
    $('dot-t2').classList.toggle('on', state.teams.team2.joined);
    const both = state.teams.team1.joined && state.teams.team2.joined;
    $('lobby-actions').classList.toggle('hidden', !both);
    $('lobby-wait').classList.toggle('hidden', both);
    const btn = $('btn-auction'), lob = state.lobby || {};
    if (lob.i_voted) { btn.textContent = 'Waiting for opponent to accept…'; btn.disabled = true; }
    else if (lob.opponent_voted) { btn.textContent = '✅ Accept Auction (opponent ready)'; btn.disabled = false; }
    else { btn.textContent = '🏏 Start Auction Draft'; btn.disabled = false; }
}

// ---------- tournament lobby ----------
function renderTournamentLobby(state) {
    const tl = state.tournament_lobby;
    $('t-lobby-code').textContent = state.code;
    $('t-lobby-count').textContent = `${tl.joined_count}/${tl.size} teams joined`;
    $('t-lobby-roster').innerHTML = tl.roster.map(r =>
        `<div class="t-roster-slot ${r.joined ? 'joined' : 'empty'}">
            <div class="tname">${r.joined ? r.name : '—'}</div>
            <div class="tstatus">${r.joined ? 'Ready' : 'Waiting…'}</div>
        </div>`).join('');
    const btn = $('btn-t-start');
    btn.classList.toggle('hidden', !tl.all_joined);
    if (tl.all_joined) {
        if (tl.i_voted) { btn.textContent = 'Waiting for everyone to accept…'; btn.disabled = true; }
        else { btn.textContent = '🏏 Start Auction Draft'; btn.disabled = false; }
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
        <div class="t-fixture-row ${f.played ? '' : 'pending'}">
            <span><span class="kind">${f.kind.replace('_', ' ')}</span> ${f.a_name} vs ${f.b_name}</span>
            <span>${f.played ? (f.result_text || '') : 'upcoming'}</span>
        </div>`).join('');
    return `<div class="t-fixtures">${rows}</div>`;
}

function renderBracket(state) {
    const t = state.tournament;
    let html = `<div class="tagline" style="text-align:center;">Tournament Bracket</div>`;
    if (t.current_fixture) {
        html += `<div class="t-current-fixture">
            <div class="tagline">Now Playing</div>
            <div>${t.current_fixture.a_name} <span class="vs">VS</span> ${t.current_fixture.b_name}</div>
            <div class="tagline" style="margin-top:0.3rem;">${t.current_fixture.kind.replace('_', ' ')}</div>
        </div>`;
    }
    html += standingsTable(t.standings, 3);
    html += fixturesList(t.fixtures);
    $('t-bracket-wrap').innerHTML = html;
}

function renderChampion(state) {
    const t = state.tournament;
    // only the two finalists actually received match/scorecard data from the
    // server (spectators are redacted to the bracket summary only)
    const canViewScorecard = !!state.match;
    $('t-bracket-wrap').innerHTML = `
        <div class="t-champion-banner">
            <div class="trophy">🏆</div>
            <h1>${t.champion_name}</h1>
            <div class="tagline">TOURNAMENT CHAMPIONS</div>
        </div>
        ${standingsTable(t.standings, 3)}
        ${fixturesList(t.fixtures)}
        <div style="display:flex; gap:0.8rem; margin-top:1rem;">
            ${canViewScorecard ? '<button class="btn-ghost btn-lg" id="btn-t-view-final">📋 View Final Scorecard</button>' : ''}
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
}

// ---------- game shell ----------
function renderGame(state) {
    const m = state.match;

    // TOSS — happens before any innings state exists
    if (m.stage === 'toss') {
        $('role-pill').textContent = '🪙 Toss';
        $('role-pill').className = 'role-pill';
        renderToss(m);
        return;
    }
    hideOverlay('toss-overlay');

    $('role-pill').textContent = m.i_am_batting ? '🏏 Batting' : '🎯 Bowling';
    $('role-pill').className = 'role-pill ' + (m.i_am_batting ? 'batting' : 'bowling');

    // reset stale bowler selection if it's no longer valid on the bench
    if (!m.i_am_batting) {
        const legal = m.my_bench.filter(b => !b.disabled).map(b => b.name);
        if (ui.selectedBowler && !legal.includes(ui.selectedBowler)) ui.selectedBowler = null;
    }
    // adopt server-locked intents once submitted (so both sides agree post-reveal)
    if (m.pending.i_submitted) {
        const mine = m.pending.mine;
        if (m.i_am_batting) {
            if (m.striker) setBatterIntent(m.striker.name, mine.striker_intent);
            if (m.non_striker) setBatterIntent(m.non_striker.name, mine.non_striker_intent);
        } else { ui.bowlIntent = mine.bowl_intent; ui.selectedBowler = mine.bowler_name; }
    }

    renderScoreboard(m);
    renderGround(m);
    renderReadyBar(m);
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
    if (m.toss.i_won) {
        card.innerHTML = `
            <div class="toss-coin">🪙</div>
            <h2>You won the toss!</h2>
            <div class="sub">${m.toss.winner_name}, what will you do?</div>
            <div class="toss-choices">
                <button class="btn-go btn-lg" data-toss="bat">Bat First</button>
                <button class="btn-red btn-lg" data-toss="bowl">Bowl First</button>
            </div>`;
        card.querySelectorAll('[data-toss]').forEach(b =>
            b.addEventListener('click', () => tossChoice(b.dataset.toss)));
    } else {
        card.innerHTML = `
            <div class="toss-coin">🪙</div>
            <h2>Coin is in the air…</h2>
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
            attrs: `data-opener="${b.name}"`, extraHtml: badge
        });
    }).join('');
    const ready = ui.openerPicks.length === 2;
    card.innerHTML = `
        <h2>Pick Your Openers</h2>
        <div class="sub">Tap two batsmen — first pick takes strike.</div>
        <div class="openers-grid">${grid}</div>
        <button class="btn-go btn-lg" id="confirm-openers" ${ready ? '' : 'disabled'}>
            ${ready ? 'Send Them Out ✔' : `Pick ${2 - ui.openerPicks.length} more`}</button>`;
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
    $('scoreboard').innerHTML = `
        <div>
            <div class="sb-team">${m.batting_team_name}</div>
            <div class="sb-score">${m.runs}/${m.wickets}</div>
            <div class="sb-meta">Overs ${m.overs} / ${m.max_overs} &middot; Extras ${m.extras}</div>
        </div>
        <div class="sb-meta">bowling: ${m.bowling_team_name}</div>
        ${target}`;
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
    return `
    <div class="${cls.join(' ')}" ${opts.attrs || ''}>
        ${opts.extraHtml || ''}
        <div class="pcard-top">
            <div class="ovr-chip"><b>${card.batting_ovr ?? '--'}</b><span>BAT</span></div>
            <div class="ovr-chip"><b>${card.bowling_ovr ?? '--'}</b><span>BOWL</span></div>
        </div>
        <div class="pcard-body">
            <div class="pname">${card.name}</div>
            ${fig}${tag}
        </div>
    </div>`;
}

function emptySlot(text, dropAttrs = '') {
    return `<div class="pcard is-empty" ${dropAttrs}>${text}</div>`;
}

function intentSlider(kind, value, disabled, batterName) {
    const batterAttr = batterName ? ` data-batter="${batterName}"` : '';
    const extremeClass = value <= 10 ? ' intent-extreme-low' : value >= 90 ? ' intent-extreme-high' : '';
    return `
    <div class="intent">
        <input type="range" min="0" max="100" step="5" value="${value}" data-intent="${kind}"${batterAttr} class="${extremeClass.trim()}" ${disabled ? 'disabled' : ''}>
        <div class="intent-word" id="word-${kind}">${intentWord(value)}</div>
    </div>`;
}

// ---------- ground ----------
function bowlerCard(cb) {
    return pcard({ name: cb.name, batting_ovr: cb.batting_ovr ?? null, bowling_ovr: cb.bowling_ovr },
        { figure: `${cb.wickets}-${cb.runs}(${cb.overs})` });
}

function renderGround(m) {
    const g = $('ground');
    const isFreeHit = m.stage === 'free_hit';
    const locked = isFreeHit ? m.free_hit.i_ready : m.pending.i_submitted;
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
            <div class="slot-label">On Strike ⭐</div>
            ${pcard(m.striker, { figure: `${m.striker.runs}(${m.striker.balls})` })}
            ${m.i_am_batting ? intentSlider('striker', getBatterIntent(m.striker.name), slDisabled, m.striker.name) : ''}
            ${retireBtn('striker')}
        </div>`;
    } else if (m.await_next_batter && m.i_am_batting) {
        strikerSlot = `<div class="ground-slot">
            <div class="slot-label">New Batsman ⬇</div>
            ${emptySlot('Drop a batsman here', 'id="drop-striker" data-drop="1"')}
        </div>`;
    } else {
        strikerSlot = `<div class="ground-slot"><div class="slot-label">On Strike</div>${emptySlot('—')}</div>`;
    }

    let nonStrikerSlot;
    if (m.non_striker) {
        nonStrikerSlot = `<div class="ground-slot">
            <div class="slot-label">Non-Striker</div>
            ${pcard(m.non_striker, { figure: `${m.non_striker.runs}(${m.non_striker.balls})` })}
            ${m.i_am_batting ? intentSlider('nonstriker', getBatterIntent(m.non_striker.name), slDisabled, m.non_striker.name) : ''}
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
        if (ob) { label = 'Bowler This Over'; cardHtml = bowlerCard(ob); }
        else if (m.current_bowler) { label = "Last Over's Bowler"; cardHtml = bowlerCard(m.current_bowler); }
        else { label = 'Bowler'; cardHtml = emptySlot('Awaiting bowler…'); }
        bowlerSlot = `<div class="ground-slot"><div class="slot-label">${label}</div>${cardHtml}</div>`;
    } else if (m.stage === 'play') {
        const sel = ui.selectedBowler ? m.my_bench.find(b => b.name === ui.selectedBowler) : null;
        bowlerSlot = `<div class="ground-slot"><div class="slot-label">Your Bowler This Over</div>` +
            (sel
                ? pcard(sel, { tag: `${sel.overs_bowled}/${sel.max_overs} overs` }) +
                intentSlider('bowl', ui.bowlIntent, slDisabled)
                : emptySlot('Pick a bowler from your bench ⬇')) +
            `</div>`;
    } else {
        bowlerSlot = `<div class="ground-slot"><div class="slot-label">Your Bowler</div>` +
            (m.current_bowler ? bowlerCard(m.current_bowler) : emptySlot('—')) +
            (isFreeHit ? intentSlider('bowl', ui.bowlIntent, slDisabled) : '') +
            `</div>`;
    }

    g.innerHTML = `<div class="ground-row batters-row">${strikerSlot}${nonStrikerSlot}</div>
                   <div class="ground-row bowler-row">${bowlerSlot}</div>`;
    wireSliders();
    wireDropZone(m);
    wireRetireButtons();
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

function wireSliders() {
    document.querySelectorAll('#ground input[data-intent]').forEach(inp => {
        let wasExtreme = inp.classList.contains('intent-extreme-low') || inp.classList.contains('intent-extreme-high');
        inp.addEventListener('input', () => {
            const v = parseInt(inp.value);
            const kind = inp.dataset.intent;
            const word = $(`word-${kind}`);
            if (word) word.textContent = intentWord(v);
            if (kind === 'striker' || kind === 'nonstriker') {
                if (inp.dataset.batter) setBatterIntent(inp.dataset.batter, v);
            } else if (kind === 'bowl') {
                ui.bowlIntent = v;
            }
            // Friction feedback: past +-10 of either end the wicket-factor swing is
            // now much stronger (strength=2.0 drives dot-ball probability to exactly
            // 0% at full aggression) -- escalate the thumb visually and give a short
            // haptic tick on entry so pushing that far reads as a heavier, more
            // consequential move rather than a free, weightless drag.
            const isLow = v <= 10, isHigh = v >= 90;
            inp.classList.toggle('intent-extreme-low', isLow);
            inp.classList.toggle('intent-extreme-high', isHigh);
            const isExtreme = isLow || isHigh;
            if (isExtreme && !wasExtreme && navigator.vibrate) navigator.vibrate(15);
            wasExtreme = isExtreme;
        });
    });
}

// ---------- ready bar ----------
function oppReadyTxt(ready) {
    return `<span class="ready-status">Opponent: ${ready
        ? '<span class="on">READY ✔</span>' : '<span class="off">setting up…</span>'}</span>`;
}

function renderReadyBar(m) {
    const bar = $('ready-bar');

    if (m.stage === 'await_batter') {
        bar.innerHTML = m.i_am_batting
            ? `<div class="ready-status"><span class="off">Send in a new batsman — tap or drag one from your bench.</span></div>`
            : `<div class="ready-status"><span class="off">Waiting for the batting side to send a new batsman…</span></div>`;
        return;
    }

    if (m.stage === 'await_resume') {
        if (m.i_am_batting) {
            bar.innerHTML = `<button class="btn-go btn-lg" id="btn-resume">Ready to Resume Over ✔</button>
                <span class="ready-status"><span class="off">New batsman is in — set intent, then resume.</span></span>`;
            $('btn-resume').addEventListener('click', submitResume);
        } else {
            bar.innerHTML = `<div class="ready-status"><span class="off">Waiting for the batting side to resume the over…</span></div>`;
        }
        return;
    }

    if (m.stage === 'free_hit') {
        const oppTxt = oppReadyTxt(m.free_hit.opponent_ready);
        if (m.free_hit.i_ready) {
            bar.innerHTML = `<button class="btn-ghost" disabled>Free hit locked — waiting…</button>${oppTxt}`;
        } else {
            bar.innerHTML = `<span class="ready-status" style="color:var(--gold)">⚡ FREE HIT!</span>
                <button class="btn-gold btn-lg" id="btn-fh">Confirm Intent ✔</button>${oppTxt}`;
            $('btn-fh').addEventListener('click', submitFreeHit);
        }
        return;
    }

    // stage === 'play' — SEQUENCED: the bowling side locks the bowler first,
    // then the batting side sees who's bowling and sets its strategy.
    if (!m.i_am_batting) {
        if (m.pending.i_submitted) {
            bar.innerHTML = `<button class="btn-ghost" disabled>Bowler locked ✔ — waiting for the batsmen…</button>`;
        } else {
            const disabled = !ui.selectedBowler;
            bar.innerHTML =
                `<button class="btn-go btn-lg" id="btn-ready" ${disabled ? 'disabled' : ''}>Lock In Bowler ✔</button>
                 <span class="ready-status"><span class="off">${disabled ? 'Pick a bowler first' : "Batsmen won't see your intent"}</span></span>`;
            const btn = $('btn-ready');
            if (btn) btn.addEventListener('click', submitOver);
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
         <button class="btn-go btn-lg" id="btn-ready">Set Strategy &amp; Ready ✔</button>`;
    $('btn-ready').addEventListener('click', submitOver);
}

async function submitOver() {
    const m = CURRENT.match;
    try {
        if (m.i_am_batting) {
            await Net.post('/api/submit_over', {
                token: Net.getToken(),
                striker_intent: getBatterIntent(m.striker.name),
                non_striker_intent: getBatterIntent(m.non_striker.name),
            });
        } else {
            if (!ui.selectedBowler) return toast('Pick a bowler first.');
            await Net.post('/api/submit_over', {
                token: Net.getToken(),
                bowler_name: ui.selectedBowler,
                bowl_intent: ui.bowlIntent,
            });
        }
        Net.forceRefresh();
    } catch (e) { toast(e.message); }
}

async function submitFreeHit() {
    const m = CURRENT.match;
    const body = m.i_am_batting
        ? { token: Net.getToken(), striker_intent: getBatterIntent(m.striker.name), non_striker_intent: getBatterIntent(m.non_striker.name) }
        : { token: Net.getToken(), bowl_intent: ui.bowlIntent };
    try { await Net.post('/api/free_hit', body); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
}

async function submitResume() {
    const m = CURRENT.match;
    try {
        await Net.post('/api/ready_resume', {
            token: Net.getToken(),
            striker_intent: getBatterIntent(m.striker.name),
            non_striker_intent: getBatterIntent(m.non_striker.name),
        });
        Net.forceRefresh();
    } catch (e) { toast(e.message); }
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

function renderThisOver(m) {
    const box = $('this-over');
    const entries = m.this_over;
    if (!entries.length) {
        box.innerHTML = '<div class="comm-empty">The over will play out here, ball by ball.</div>';
        overAnim.key = null;
        overAnim.shown = 0;
        return;
    }
    // key changes when a fresh over starts (server clears this_over each over)
    const key = m.innings + ':' + entries[0].ball;
    if (key !== overAnim.key) {
        overAnim.key = key;
        overAnim.shown = 0;
        box.innerHTML = '';
    }
    // reveal only the not-yet-shown deliveries, one after another
    for (let i = overAnim.shown; i < entries.length; i++) {
        const delay = (i - overAnim.shown) * 130;
        const entry = entries[i];
        setTimeout(() => {
            box.insertAdjacentHTML('beforeend', commLine(entry));
            box.scrollTop = box.scrollHeight;
        }, delay);
    }
    overAnim.shown = entries.length;
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
    const wrap = $('sc-wrap');
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
            <h3><span>${inn.batting_team_name}${inn.in_progress ? ' (batting)' : ''}</span>
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
function swingGavel() {
    const g = $('gavel'); if (!g) return;
    g.classList.remove('swing'); void g.offsetWidth; g.classList.add('swing');
}
let aucFx = { strike: -1, stage: null };
function maybeGavel(a) {
    const struck = a.stage === 'bidding' && a.strike > aucFx.strike && a.strike > 0;
    const resolved = a.stage === 'resolved' && aucFx.stage !== 'resolved';
    if (struck || resolved) { swingGavel(); playGavel(resolved); }
    aucFx = { strike: a.stage === 'bidding' ? a.strike : 0, stage: a.stage };
}

function auctioneerScene() {
    return `<div class="auctioneer-scene">
        <div class="auctioneer" title="Auctioneer">🧑‍⚖️</div>
        <div class="gavel-wrap"><span class="gavel" id="gavel">🔨</span><span class="gavel-block"></span></div>
    </div>`;
}

function renderAuction(state) {
    const a = state.auction;
    const me = a.you_role;
    $('auc-role').textContent = 'You: ' + (a.my_squad ? a.my_squad.name : me);
    $('auc-role').className = 'role-pill batting';
    if ($('auc-set')) {
        $('auc-set').textContent = a.set_id ? `Set ${a.set_id}: ${a.tier} ${a.role_name}s` : 'Auction';
    }

    $('auc-my').className = 'squad-panel me';
    $('auc-opp').className = 'squad-panel opp';
    $('auc-my').innerHTML = squadPanel(a.my_squad, true, a);
    $('auc-opp').innerHTML = a.opp_squad
        ? squadPanel(a.opp_squad, false, a)
        : otherSquadsPanel(a.other_squads, a);
    $('auc-center').innerHTML = auctionCenter(a, me);
    wireAuctionCenter(a, me);

    // Always render full pool into the new tab
    if ($('auc-full-pool')) {
        const sold = [
            ...(a.my_squad ? a.my_squad.roster.map(p => p.name) : []),
            ...(a.opp_squad ? a.opp_squad.roster.map(p => p.name) : []),
            ...(a.other_squads || []).flatMap(s => s.roster_names || []),
        ];
        $('auc-full-pool').innerHTML = poolList(a.pool, sold);
    }

    if (a.stage === 'bidding' || a.stage === 'done') setAucTimer(a.time_left_ms, a.total_wait_ms);
    else setAucTimer(0, 0);

    maybeGavel(a);
}

function squadPanel(sq, isMe, a) {
    const rows = sq.roster.map(p =>
        `<div class="roster-item"><span>${p.name} ${p.is_foreigner ? '<span class="os">OS</span>' : ''}</span>
         <span class="price">₹${(typeof p.price === 'number' ? p.price.toFixed(1) : (p.price ?? 0))}</span></div>`).join('')
        || '<div class="bench-empty">No buys yet.</div>';
    const pct = Math.min(100, Math.round((sq.count / a.squad_min) * 100));
    const need = Math.max(0, a.squad_min - sq.count);
    const progLabel = sq.count >= a.squad_min
        ? (sq.wk >= 1 ? 'Squad legal — can lock in ✔' : 'Need a wicket-keeper')
        : `${need} more to reach the minimum ${a.squad_min}`;
    return `<h3>${sq.name}${isMe ? ' (You)' : ''} ${sq.locked ? '<span class="badge-tier">LOCKED</span>' : ''}</h3>
        <div class="purse">₹${sq.budget.toFixed(1)} Cr</div>
        <div class="squad-stats">
            <span>Squad <b>${sq.count}</b>/${a.squad_max}</span>
            <span>Overseas <b>${sq.os}</b></span>
            <span>Keepers <b>${sq.wk}</b></span>
        </div>
        <div class="squad-progress"><div class="squad-progress-fill" style="width:${pct}%"></div></div>
        <div class="squad-progress-label">${progLabel}</div>
        ${rows}`;
}

function teamName(a, role) {
    if (role === a.you_role) return a.my_squad.name;
    if (a.opp_squad && a.opp_squad.team_id === role) return a.opp_squad.name;
    const found = (a.other_squads || []).find(s => s.team_id === role);
    return found ? found.name : role;
}

function otherSquadsPanel(others, a) {
    if (!others || !others.length) return '<h3>Other Squads</h3><div class="bench-empty">—</div>';
    const rows = others.map(sq => `
        <div class="roster-item" style="align-items:center;">
            <span>${sq.name} ${sq.locked ? '<span class="badge-tier">LOCKED</span>' : ''}</span>
            <span class="price">₹${sq.budget.toFixed(1)} Cr &middot; ${sq.count}/${a.squad_max} &middot; 🧤${sq.wk}</span>
        </div>`).join('');
    return `<h3>Other Squads</h3>${rows}`;
}

function instructionsHtml(a) {
    return `<div class="auc-instructions">
        <b>Squad rules:</b> ${a.squad_min}–${a.squad_max} players &middot; at least 1 wicket-keeper.
        <span class="note">Note: you can't keep more than ${a.xi_max_os} overseas players in your Playing XI.</span></div>`;
}

function poolList(pool, soldNames = []) {
    if (!pool) return '';
    const bySet = {};
    pool.forEach(p => { (bySet[p.set_id] = bySet[p.set_id] || { tier: p.tier, role: p.role, players: [] }).players.push(p); });
    const blocks = Object.keys(bySet).map(id => {
        const s = bySet[id];
        const items = s.players.map(p => {
            const isSold = soldNames.includes(p.name);
            return `<div class="pool-item ${isSold ? 'sold' : ''}" style="${isSold ? 'opacity:0.4; text-decoration:line-through; font-style:italic;' : ''}">${p.name} ${p.is_foreigner ? '✈' : ''} ${p.is_keeper ? '🧤' : ''}
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
        statusHtml = `Opponent: ${oppReady ? 'ready ✔' : 'not ready yet'}`;
    } else {
        const others = a.other_squads || [];
        const readyCount = others.filter(s => a.ready[s.team_id] || s.locked).length;
        statusHtml = `Other teams ready: ${readyCount}/${others.length}`;
    }
    return `<button class="btn-go btn-lg" id="auc-ready" ${a.ready[me] ? 'disabled' : ''}>
                ${a.ready[me] ? 'Ready ✔' : label}</button>
            <div class="auc-holder">${statusHtml}</div>`;
}

function lockBlock(a) {
    if (a.my_locked) return `<div class="auc-holder">🔒 Your squad is LOCKED — waiting for the others.</div>`;
    if (a.can_lock) return `<button class="btn-gold btn-lg" id="auc-lock">🔒 Lock In Squad (${a.my_squad.count} players)</button>`;
    return '';
}

function auctionCenter(a, me) {
    const instr = instructionsHtml(a);

    if (a.stage === 'preview') {
        const sold1 = a.my_squad ? a.my_squad.roster.map(p => p.name) : [];
        const sold2 = a.opp_squad ? a.opp_squad.roster.map(p => p.name) : [];
        return instr + `<div class="auc-msg">${a.message}</div>`
            + readyBlock(a, me, "I'm Ready to Bid") + poolList(a.pool, [...sold1, ...sold2]);
    }

    if (a.stage === 'bidding') {
        const c = a.current;
        const tier = c.is_foreigner ? '<span class="badge-tier badge-os">OVERSEAS</span>' : '';
        const card = pcard(c, { ovr: Math.max(c.batting_ovr, c.bowling_ovr), tag: `${a.tier} ${a.role_name}` });
        let holder = 'No bids yet — opening price.';
        if (a.active_bidder) holder = a.active_bidder === me ? '⭐ You hold the top bid' : `${teamName(a, a.active_bidder)} holds the bid`;
        const strikeTxt = a.strike === 1 ? 'GOING ONCE…' : a.strike === 2 ? 'GOING TWICE…' : '';
        const controls = (a.out[me] || a.my_locked)
            ? `<div class="auc-holder">${a.my_locked ? 'Your squad is locked — sitting out.' : 'You pulled out of this lot.'}</div>`
            : `<div class="bid-controls">
                 <div class="quick-adds">
                   <button data-bid="0.5">+0.5</button><button data-bid="1">+1</button>
                   <button data-bid="2">+2</button><button data-bid="5">+5</button>
                 </div>
                 <div class="bid-row">
                   <input type="number" id="bid-custom" step="0.1" min="0" placeholder="custom +" style="width:110px;">
                   <button class="btn-go" id="bid-send">Bid</button>
                   <button class="btn-red" id="bid-out">I'm Out</button>
                 </div>
               </div>`;
        return auctioneerScene() + `<div class="auc-msg">${a.message}</div>
            <div class="auc-timer"><div class="auc-timer-fill" id="auc-timer-fill"></div></div>
            <div style="text-align:center">${tier}</div>
            <div class="auc-player big-card">${card}</div>
            <div class="auc-bid">₹${a.current_bid.toFixed(1)} Cr</div>
            <div class="auc-holder">${holder}</div>
            <div class="auc-msg" style="color:var(--leather); min-height:1rem">${strikeTxt}</div>
            ${controls}
            ${lockBlock(a)}`;
    }

    if (a.stage === 'resolved') {
        const col = a.last_result === 'sold' ? 'var(--green-go)' : 'var(--leather)';
        return instr + auctioneerScene()
            + `<div class="auc-msg" style="font-size:1.5rem; color:${col}">${a.message}</div>`
            + lockBlock(a)
            + readyBlock(a, me, 'Ready for Next Lot');
    }

    // done
    const sq = a.my_squad;
    let hint = '';
    if (sq.count < a.squad_min) hint = `Need at least ${a.squad_min} players — use Auto-Fill.`;
    else if (sq.wk < 1) hint = 'Need at least 1 wicket-keeper.';
    const secsLeft = Math.max(0, Math.ceil((a.time_left_ms || 0) / 1000));
    const forfeitWarning = a.opp_squad ? 'or you auto-forfeit!' : 'or we auto-fill it for you!';
    const countdown = !a.my_locked
        ? `<div class="auc-holder" style="color:${secsLeft <= 15 ? 'var(--leather)' : 'var(--gold)'}">
             ⏱ ${secsLeft}s to lock a valid squad ${forfeitWarning}</div>
           <div class="auc-timer"><div class="auc-timer-fill" id="auc-timer-fill"></div></div>`
        : '';
    const othersStatus = a.opp_squad
        ? `Opponent: ${a.opp_locked ? 'locked ✔' : 'still building…'}`
        : `Others locked: ${(a.other_squads || []).filter(s => s.locked).length}/${(a.other_squads || []).length}`;
    return instr + `<div class="auc-msg">${a.message}</div>
        ${countdown}
        <div class="auc-holder">Your squad: ${sq.count}/${a.squad_max} · keepers ${sq.wk} · overseas ${sq.os}</div>
        ${sq.count < a.squad_min ? `<button class="btn-gold btn-lg" id="auc-autofill">Auto-Fill to ${a.squad_min}</button>` : ''}
        ${lockBlock(a)}
        <div class="auc-holder">${hint}</div>
        <div class="auc-holder">${othersStatus}</div>`;
}

function wireAuctionCenter(a, me) {
    const ready = $('auc-ready'); if (ready) ready.addEventListener('click', () => aucAction('/api/auction_ready'));
    document.querySelectorAll('#auc-center [data-bid]').forEach(b =>
        b.addEventListener('click', () => aucAction('/api/bid', { amount: parseFloat(b.dataset.bid) })));
    const send = $('bid-send');
    if (send) send.addEventListener('click', () => {
        const v = parseFloat(($('bid-custom') || {}).value) || 0;
        aucAction('/api/bid', { amount: v });
    });
    const out = $('bid-out'); if (out) out.addEventListener('click', () => aucAction('/api/pull_out'));
    const af = $('auc-autofill'); if (af) af.addEventListener('click', () => aucAction('/api/auto_fill'));
    const lk = $('auc-lock'); if (lk) lk.addEventListener('click', () => aucAction('/api/lock_squad'));
}

async function aucAction(path, body = {}) {
    ensureAudio();   // unlock WebAudio on the first user gesture
    try { await Net.post(path, { token: Net.getToken(), ...body }); Net.forceRefresh(); }
    catch (e) { toast(e.message); }
}

// ============================================================
//  XI SELECTION
// ============================================================
function renderXI(state) {
    const x = state.xi;
    $('xi-role').textContent = 'You: ' + x.team_name;
    $('xi-role').className = 'role-pill batting';

    const osBad = x.os > x.max_os, wkBad = x.wk < 1, cntOk = x.count === x.size;
    const bench = x.roster.filter(p => !p.selected);
    const chosen = x.roster.filter(p => p.selected);
    const othersStatus = x.opponent_locked !== null
        ? `Opponent: ${x.opponent_locked ? 'locked ✔' : 'selecting…'}`
        : `Others locked: ${x.others_locked_count}/${x.others_total}`;

    $('xi-main').innerHTML = `
        <div class="xi-stats">
            <span class="${cntOk ? 'good' : ''}">Selected ${x.count}/${x.size}</span>
            <span class="${osBad ? 'bad' : ''}">Overseas ${x.os}/${x.max_os}</span>
            <span class="${wkBad ? 'bad' : 'good'}">Keepers ${x.wk}</span>
            ${x.locked ? '<span class="good">LOCKED ✔</span>' : ''}
            <span style="margin-left:auto">${othersStatus}</span>
        </div>
        <div class="xi-cols">
            <div class="xi-col"><h4>Squad Bench (${bench.length})</h4><div class="xi-list" id="xi-bench"></div></div>
            <div class="xi-col"><h4>Playing XI (${chosen.length}/${x.size})</h4><div class="xi-list" id="xi-chosen"></div></div>
        </div>
        <button class="btn-go btn-lg" id="xi-lock" ${(cntOk && !osBad && !wkBad && !x.locked) ? '' : 'disabled'}>
            ${x.locked ? 'XI Locked — waiting for the others…' : 'Lock In XI'}</button>`;

    $('xi-bench').innerHTML = bench.map(p => xiCard(p, x)).join('') || '<div class="bench-empty">—</div>';
    $('xi-chosen').innerHTML = chosen.map(p => xiCard(p, x)).join('') || '<div class="bench-empty">Tap players to add them</div>';

    if (!x.locked) {
        document.querySelectorAll('#xi-main [data-xi]').forEach(el =>
            el.addEventListener('click', () => aucAction('/api/toggle_xi', { player_name: el.dataset.xi })));
    }
    const lk = $('xi-lock'); if (lk) lk.addEventListener('click', () => aucAction('/api/lock_xi'));
}

function xiCard(p, x) {
    const isKeeper = p.is_keeper || p.assigned_role === 'Wicket Keeper';
    return pcard(p, {
        ovr: Math.max(p.batting_ovr || 0, p.bowling_ovr || 0),
        selectable: !x.locked, selected: p.selected,
        attrs: !x.locked ? `data-xi="${p.name}"` : '',
        tag: `${p.is_foreigner ? 'OS · ' : ''}${isKeeper ? 'WK' : (p.assigned_role || '')}`,
    });
}

// ---------- boot ----------
if (Net.getToken()) Net.startPolling();
