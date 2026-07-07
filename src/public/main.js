let state = {
    teams: [
        { id: 't1', name: 'Team 1', budget: 100.0, roster: [], os: 0, wk: 0, out: false, locked: false, submitted: false, xi: [], xiLocked: false },
        { id: 't2', name: 'Team 2', budget: 100.0, roster: [], os: 0, wk: 0, out: false, locked: false, submitted: false, xi: [], xiLocked: false }
    ],
    draftSets: [],
    unsoldQueue: [],

    currentSetIndex: 0,
    currentPlayerIndex: 0,

    activeBidder: null,
    currentBid: 0.0,

    auctioneer: {
        strikeLevel: 0,
        timeRemainingMs: 0,
        totalWaitMs: 1000,
        timer: null
    },

    t1Ready: false,
    t2Ready: false,

    t1Auto: false,
    t2Auto: false
};

// --- START ---
async function initAuction() {
    const t1Name = document.getElementById('t1-name').value || 'Team 1';
    const t2Name = document.getElementById('t2-name').value || 'Team 2';

    state.teams[0].name = t1Name;
    state.teams[1].name = t2Name;

    document.getElementById('t1-display-name').innerText = t1Name;
    document.getElementById('t2-display-name').innerText = t2Name;
    document.getElementById('btn-t1-ready').innerText = t1Name.toUpperCase() + " READY";
    document.getElementById('btn-t2-ready').innerText = t2Name.toUpperCase() + " READY";

    document.getElementById('btn-start-auction').style.display = 'none';
    document.getElementById('loader').style.display = 'block';

    try {
        const res = await fetch('/api/draft');
        const data = await res.json();
        if (data.status === 'success') {
            state.draftSets = data.sets;
            document.getElementById('setup-screen').style.display = 'none';
            document.getElementById('auction-screen').style.display = 'flex';
            loadPlayer();
        } else {
            showModal("Error", data.message);
        }
    } catch (err) {
        showModal("Error", "Could not connect to backend engine.");
    }
}

// --- PLAYER LOADING & READY PRE-CHECKS ---
function loadPlayer() {
    if (state.currentSetIndex >= state.draftSets.length) {
        showModal("Auction Concluded", "All sets exhausted! Review your Unsold Queue or use the final Submit bounds.");
        return;
    }
    const currentSet = state.draftSets[state.currentSetIndex];
    if (state.currentPlayerIndex >= currentSet.players.length) {
        state.currentSetIndex++;
        state.currentPlayerIndex = 0;
        loadPlayer();
        return;
    }
    if (state.currentPlayerIndex === 0) {
        displayReadyCheck('set');
    } else {
        displayReadyCheck('player');
    }
}

function displayReadyCheck(type) {
    state.t1Ready = false;
    state.t2Ready = false;
    document.getElementById('btn-t1-ready').classList.remove('active');
    document.getElementById('btn-t2-ready').classList.remove('active');

    const nextSet = state.draftSets[state.currentSetIndex];
    if (type === 'set') {
        document.getElementById('break-title').textContent = `Prepare for Set ${nextSet.set_id}`;
        document.getElementById('break-subtitle').textContent = `${nextSet.tier} ${nextSet.role}s`;
    } else {
        document.getElementById('break-title').textContent = `Next Player...`;
        document.getElementById('break-subtitle').textContent = `Continuing Set ${nextSet.set_id}`;
    }

    document.getElementById('auctioneer-box').style.display = 'none';
    document.getElementById('player-card').style.display = 'none';
    document.getElementById('bidding-arena').style.display = 'none';
    document.getElementById('set-break-screen').style.display = 'flex';
}

function teamReady(tid) {
    if (tid === 't1') {
        state.t1Ready = true;
        document.getElementById('btn-t1-ready').classList.add('active');
    }
    if (tid === 't2') {
        state.t2Ready = true;
        document.getElementById('btn-t2-ready').classList.add('active');
    }
    if (state.t1Ready && state.t2Ready) {
        document.getElementById('set-break-screen').style.display = 'none';
        document.getElementById('auctioneer-box').style.display = 'flex';
        document.getElementById('player-card').style.display = 'block';
        document.getElementById('bidding-arena').style.display = 'block';
        executeLoadPlayer();
    }
}

function executeLoadPlayer() {
    const currentSet = state.draftSets[state.currentSetIndex];
    const player = currentSet.players[state.currentPlayerIndex];
    document.getElementById('set-indicator').textContent = `Set ${currentSet.set_id}: ${currentSet.tier} ${currentSet.role}s`;
    document.getElementById('player-name').textContent = player.name;
    document.getElementById('player-role').textContent = currentSet.role;

    if (player.is_foreigner) {
        document.getElementById('player-os').textContent = "OVERSEAS";
        document.getElementById('player-os').classList.add("active");
    } else {
        document.getElementById('player-os').textContent = "LOCAL";
        document.getElementById('player-os').classList.remove("active");
    }
    const initials = player.name.split(" ").map(n => n[0]).join("");
    document.getElementById('player-avatar').src = `https://ui-avatars.com/api/?name=${initials}&background=0D8ABC&color=fff&size=128`;
    document.getElementById('player-bat').textContent = player.batting_ovr || 55;
    document.getElementById('player-bowl').textContent = player.bowling_ovr || 55;

    state.activeBidder = null;
    state.currentBid = 0.5;
    state.teams[0].out = false;
    state.teams[1].out = false;
    document.getElementById("t1-draft-val").value = 0.0;
    document.getElementById("t2-draft-val").value = 0.0;

    updateBidUI();
    state.auctioneer.strikeLevel = 0;
    setAuctioneerTimer("Opening at 50 Lakhs. Any bids?");
}

// --- AUCTIONEER LOGIC ---
function getTierBaseWait() {
    const tier = state.draftSets[state.currentSetIndex].tier;
    if (tier === "Marquee") return 5000;
    if (tier === "Mid-Level") return 3500;
    return 2000;
}

function setAuctioneerTimer(customText) {
    if (state.auctioneer.timer) clearInterval(state.auctioneer.timer);
    let rand = getTierBaseWait() * (0.8 + Math.random() * 0.4);
    state.auctioneer.timeRemainingMs = rand;
    state.auctioneer.totalWaitMs = rand;
    if (customText) document.getElementById("auctioneer-text").textContent = customText;
    state.auctioneer.timer = setInterval(tickAuctioneer, 50);
}

function tickAuctioneer() {
    state.auctioneer.timeRemainingMs -= 50;
    let pct = Math.max(0, (state.auctioneer.timeRemainingMs / state.auctioneer.totalWaitMs) * 100);
    document.getElementById("timer-fill").style.width = `${pct}%`;
    document.getElementById("timer-text").textContent = `${Math.max(0, (state.auctioneer.timeRemainingMs / 1000)).toFixed(1)}s`;
    if (state.auctioneer.timeRemainingMs <= 0) processStrike();
}

function processStrike() {
    state.auctioneer.strikeLevel++;
    const currentPrice = state.currentBid.toFixed(1);
    if (state.auctioneer.strikeLevel === 1) {
        if (state.activeBidder) setAuctioneerTimer(`Fair warning... Going ONCE at ₹${currentPrice} Cr!`);
        else setAuctioneerTimer(`No bids yet... Going ONCE at base price!`);
    } else if (state.auctioneer.strikeLevel === 2) {
        if (state.activeBidder) setAuctioneerTimer(`Going TWICE! Final chance!`);
        else setAuctioneerTimer(`Going TWICE! Any takers before he's Unsold?`);
    } else {
        clearInterval(state.auctioneer.timer);
        if (state.activeBidder) executeSale();
        else executeUnsold();
    }
}

// --- BIDDING CONTROLS ---
function updateBidUI() {
    document.getElementById('current-bid-val').textContent = state.currentBid.toFixed(1);
    const bidderText = state.activeBidder ? state.teams.find(t => t.id === state.activeBidder).name + " holds!" : "No active bids";
    document.getElementById('active-bidder-name').textContent = bidderText;
}

function addDraftBid(tid, amount) {
    let el = document.getElementById(`${tid}-draft-val`);
    el.value = (parseFloat(el.value || 0) + amount).toFixed(1);
}

function submitBid(tid) {
    const teamIndex = tid === 't1' ? 0 : 1;
    const team = state.teams[teamIndex];
    if (team.out) return showModal("Locked Out", "You already pulled out of this round!");
    if (team.submitted || team.locked) return showModal("Locked", "You have locked your team actions!");

    const draftInput = document.getElementById(`${tid}-draft-val`);
    let manualBid = parseFloat(draftInput.value || 0);
    let nextBid = state.currentBid + manualBid;

    if (manualBid === 0.0) nextBid = state.currentBid + 0.5;
    if (tid === state.activeBidder && manualBid === 0.0) return;
    if (team.budget < nextBid) return showModal("Insufficient Funds", `${team.name} cannot afford ₹${nextBid.toFixed(1)} Cr!`);

    state.activeBidder = tid;
    state.currentBid = nextBid;
    state.teams[0].out = false;
    state.teams[1].out = false;
    draftInput.value = 0.0;
    updateBidUI();

    state.auctioneer.strikeLevel = 0;
    document.getElementById("auctioneer-text").textContent = `Fierce bidding! New bid by ${team.name} at ₹${nextBid.toFixed(1)} Cr!`;
    setAuctioneerTimer(null);
}

function pullOut(tid) {
    const oppTid = tid === 't1' ? 't2' : 't1';
    const teamIndex = tid === 't1' ? 0 : 1;
    state.teams[teamIndex].out = true;
    if (state.activeBidder === oppTid) {
        clearInterval(state.auctioneer.timer);
        return executeSale();
    }
    if (state.teams[0].out && state.teams[1].out) {
        clearInterval(state.auctioneer.timer);
        executeUnsold();
    }
}

// --- RESOLUTION ---
function executeSale() {
    const teamIndex = state.activeBidder === 't1' ? 0 : 1;
    const team = state.teams[teamIndex];
    const player = state.draftSets[state.currentSetIndex].players[state.currentPlayerIndex];
    const role = state.draftSets[state.currentSetIndex].role;

    document.getElementById("auctioneer-text").textContent = `SOLD! To ${team.name} for ₹${state.currentBid.toFixed(1)} Cr!`;
    if (team.roster.length >= 21) {
        showModal("Squad Full", `${team.name} has max 21 players! Canceling sale...`);
        return executeUnsold();
    }

    team.budget -= state.currentBid;
    team.roster.push({ ...player, assigned_role: role });
    if (player.is_foreigner) team.os++;
    if (player.is_keeper || role === 'Wicket Keeper') team.wk++;

    syncTeamPanel(teamIndex);
    setTimeout(() => {
        state.currentPlayerIndex++;
        loadPlayer();
    }, 3500);
}

function executeUnsold() {
    document.getElementById("auctioneer-text").textContent = `UNSOLD! Moving player to the reserve pile...`;
    const currentSet = state.draftSets[state.currentSetIndex];
    const player = currentSet.players[state.currentPlayerIndex];
    state.unsoldQueue.push(player);
    document.getElementById('unsold-count').textContent = state.unsoldQueue.length;

    let targetListId = 'unsold-' + currentSet.role;
    const list = document.getElementById(targetListId);
    if (list) {
        const badge = document.createElement('div');
        badge.className = 'unsold-badge';
        badge.textContent = `${player.name} (${player.batting_ovr || 55}|${player.bowling_ovr || 55})`;
        list.appendChild(badge);
    }
    setTimeout(() => {
        state.currentPlayerIndex++;
        loadPlayer();
    }, 3500);
}

// --- AUTO DRAFT ALGORITHM ---
function openAutoDraftModal() {
    showModal("⚡ Instant Auto-Draft", `
        <p>Which team would like to propose a mutual Auto-Draft to 15 players?</p>
        <div style="display:flex; justify-content:space-around; margin-top:2rem;">
            <button class="btn-primary" onclick="requestAutoDraft('t1')">${state.teams[0].name.toUpperCase()} PROPOSE</button>
            <button class="btn-primary" onclick="requestAutoDraft('t2')">${state.teams[1].name.toUpperCase()} PROPOSE</button>
        </div>
    `, true);
}

function requestAutoDraft(requester) {
    const oppTid = requester === 't1' ? 't2' : 't1';
    const reqName = state.teams[requester === 't1' ? 0 : 1].name;
    const oppName = state.teams[oppTid === 't1' ? 0 : 1].name;

    showModal("⚡ Auto-Draft Proposed", `
        <p><strong>${reqName.toUpperCase()}</strong> has formally requested an emergency Auto-Draft for the remaining players!</p>
        <p><strong>${oppName.toUpperCase()}</strong>, do you accept this proposal to instantly complete both squads?</p>
        <div style="display:flex; justify-content:space-around; margin-top:2rem;">
            <button class="btn-success" onclick="acceptAutoDraft()">ACCEPT & AUTO-DRAFT</button>
            <button class="btn-danger" onclick="closeModal()">DECLINE</button>
        </div>
    `, true);
}

function acceptAutoDraft() {
    closeModal();
    executeAutoDraft();
}

function executeAutoDraft() {
    let pool = [];
    state.draftSets.forEach(s => {
        s.players.forEach(p => {
            const inT1 = state.teams[0].roster.some(x => x.name === p.name);
            const inT2 = state.teams[1].roster.some(x => x.name === p.name);
            if (!inT1 && !inT2) pool.push({ ...p, assigned_role: s.role });
        });
    });

    const score = (p) => (p.batting_ovr || 55) + (p.bowling_ovr || 55);
    let keepers = pool.filter(p => p.is_keeper || p.assigned_role === 'Wicket Keeper').sort((a, b) => score(b) - score(a));
    let others = pool.filter(p => !(p.is_keeper || p.assigned_role === 'Wicket Keeper')).sort((a, b) => score(b) - score(a));

    if (state.teams[0].wk < 1 && keepers.length > 0) processAutoPick(0, keepers.shift());
    if (state.teams[1].wk < 1 && keepers.length > 0) processAutoPick(1, keepers.shift());

    pool = [...keepers, ...others].sort((a, b) => score(b) - score(a));

    let turn = 0;
    while (pool.length > 0) {
        const t0Needs = 15 - state.teams[0].roster.length;
        const t1Needs = 15 - state.teams[1].roster.length;
        if (t0Needs <= 0 && t1Needs <= 0) break;

        let active = turn % 2;
        if (active === 0 && t0Needs <= 0) active = 1;
        if (active === 1 && t1Needs <= 0) active = 0;

        const pickIndex = Math.floor(Math.random() * Math.min(3, pool.length));
        const player = pool.splice(pickIndex, 1)[0];

        processAutoPick(active, player);
        turn++;
    }

    syncTeamPanel(0);
    syncTeamPanel(1);
    showModal("Auto-Draft Complete!", "A balanced algorithm has populated your remaining roster spots natively to 15! You may now Lock and Submit!");
}

function processAutoPick(teamIdx, player) {
    if (!player) return;
    const team = state.teams[teamIdx];
    team.roster.push(player);
    if (player.is_foreigner) team.os++;
    if (player.is_keeper || player.assigned_role === 'Wicket Keeper') team.wk++;
    team.budget -= 2.0;
}

function syncTeamPanel(teamIdx) {
    const team = state.teams[teamIdx];
    document.getElementById(`${team.id}-budget`).textContent = team.budget.toFixed(1);
    document.getElementById(`${team.id}-squad-count`).textContent = team.roster.length;
    document.getElementById(`${team.id}-os-count`).textContent = team.os;
    document.getElementById(`${team.id}-wk-count`).textContent = team.wk;

    const rosterDiv = document.getElementById(`${team.id}-roster`);
    rosterDiv.innerHTML = '';
    team.roster.forEach(p => {
        const item = document.createElement('div');
        item.className = 'roster-item';
        item.innerHTML = `<span>${p.name}</span>`;
        rosterDiv.appendChild(item);
    });
}

function openUpcomingModal() {
    let html = '';
    let count = 0;
    state.draftSets.forEach((set, idx) => {
        if (idx < state.currentSetIndex) return;
        let activePlayers = set.players;
        if (idx === state.currentSetIndex) activePlayers = set.players.slice(state.currentPlayerIndex);
        if (activePlayers.length === 0) return;

        html += `<h4 style="color:var(--primary); margin-top:1.5rem; font-family:'Outfit'">Set ${set.set_id}: ${set.tier} ${set.role}s</h4><ul>`;
        activePlayers.forEach(p => {
            count++;
            html += `<li><span><strong>${p.name}</strong> ${p.is_foreigner ? '✈️' : ''}</span><span style="color:var(--success)">BAT: ${p.batting_ovr || 55} | BOWL: ${p.bowling_ovr || 55}</span></li>`;
        });
        html += `</ul>`;
    });
    document.getElementById('upcoming-global-title').textContent = `Available Players Draft Pool (${count} remaining)`;
    document.getElementById('upcoming-list-container').innerHTML = html || '<p>No more players in the draft pool.</p>';
    document.getElementById('upcoming-modal').style.display = 'flex';
}

function closeUpcomingModal() { document.getElementById('upcoming-modal').style.display = 'none'; }
function closeModal() { document.getElementById('modal-overlay').style.display = 'none'; }
function showModal(title, msg, hideOk = false) {
    document.getElementById('modal-title').textContent = title;
    document.getElementById('modal-text').innerHTML = msg;
    document.getElementById('modal-close-btn').style.display = hideOk ? 'none' : 'inline-block';
    document.getElementById('modal-overlay').style.display = 'flex';
}

// --- TEAM LOCKING & SUBMISSION ---
function lockTeam(tid) {
    const teamIndex = tid === 't1' ? 0 : 1;
    const team = state.teams[teamIndex];
    let errors = [];
    if (team.roster.length < 15 || team.roster.length > 21) errors.push(`Squad size must be 15-21 (Current: ${team.roster.length}).`);
    if (team.wk < 1) errors.push(`You need securely at least 1 Wicket Keeper.`);
    if (errors.length > 0) return showModal(`${team.name} Cannot Lock`, errors.join("<br>"));

    team.locked = true;
    const lockBtn = document.getElementById(`btn-${tid}-lock`);
    lockBtn.textContent = 'LOCKED ✔';
    lockBtn.classList.remove('btn-warning');
    lockBtn.classList.add('btn-success');
    document.getElementById(`btn-${tid}-submit`).disabled = false;
}

function submitTeam(tid) {
    const teamIndex = tid === 't1' ? 0 : 1;
    const team = state.teams[teamIndex];
    if (!team.locked) return;
    team.submitted = true;
    document.getElementById(`btn-${tid}-submit`).textContent = 'SUBMITTED ✔';

    if (state.teams[0].submitted && state.teams[1].submitted) {
        showModal("Draft Complete!", "🎉 Both squads submitted perfectly! Transitioning to Playing XI...", true);
        setTimeout(() => {
            closeModal();
            initPlayingXI();
        }, 2000);
    }
}

function launchOpenersModal(isSecondInnings = false) {
    const battingTID = isSecondInnings ? 1 : 0;
    const team = state.teams[battingTID];

    document.getElementById('openers-subtitle').textContent = `${team.name}, select who will open the batting.`;

    const sSelect = document.getElementById('striker-select');
    const nsSelect = document.getElementById('nonstriker-select');

    sSelect.innerHTML = '';
    nsSelect.innerHTML = '';

    team.xi.forEach(p => {
        const opt1 = document.createElement('option');
        opt1.value = p.name; opt1.textContent = `${p.name} (BAT: ${p.batting_ovr})`;
        sSelect.appendChild(opt1);

        const opt2 = document.createElement('option');
        opt2.value = p.name; opt2.textContent = `${p.name} (BAT: ${p.batting_ovr})`;
        nsSelect.appendChild(opt2);
    });

    nsSelect.selectedIndex = 1;

    document.getElementById('openers-modal').style.display = 'flex';
    document.getElementById('openers-modal').dataset.isInnings2 = isSecondInnings;
}

async function submitOpeners() {
    const sSelect = document.getElementById('striker-select').value;
    const nsSelect = document.getElementById('nonstriker-select').value;

    if (sSelect === nsSelect) return showModal("Invalid Pair", "Striker and Non-Striker must be different players.");

    document.getElementById('openers-modal').style.display = 'none';
    const isSecondInnings = document.getElementById('openers-modal').dataset.isInnings2 === 'true';
    const battingTID = isSecondInnings ? 1 : 0;

    // Rearrange Javascript Array so Python instantly recognizes index 0 and 1
    let lineup = state.teams[battingTID].xi;
    let strikerIndex = lineup.findIndex(p => p.name === sSelect);
    let temp1 = lineup[0]; lineup[0] = lineup[strikerIndex]; lineup[strikerIndex] = temp1;

    let nonStrikerIndex = lineup.findIndex(p => p.name === nsSelect);
    let temp2 = lineup[1]; lineup[1] = lineup[nonStrikerIndex]; lineup[nonStrikerIndex] = temp2;

    matchData.usedBatters.push(sSelect);
    matchData.usedBatters.push(nsSelect);

    if (isSecondInnings) {
        await executeStartSecondInnings();
    } else {
        await initMatchBackend();
    }
}

async function initMatchBackend() {
    showModal("Initializing Simulator", "Validating Lineups and spinning up Math Engine...", true);
    try {
        const res = await fetch('/api/init_match', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                t1_name: state.teams[0].name,
                t2_name: state.teams[1].name,
                t1_xi: state.teams[0].xi,
                t2_xi: state.teams[1].xi
            })
        });
        const data = await res.json();
        if (data.status === 'success') {
            closeModal();
            initMatchHUD();
        } else {
            showModal("Engine Error", data.message);
        }
    } catch (err) {
        showModal("Connection Error", "Failed to reach Engine Server.");
    }
}

// --- PLAYING XI SCREEN LOGIC ---
function initPlayingXI() {
    document.getElementById('auction-screen').style.display = 'none';
    document.getElementById('playing-xi-screen').style.display = 'flex';
    document.getElementById('xi-t1-name').textContent = state.teams[0].name;
    document.getElementById('xi-t2-name').textContent = state.teams[1].name;
    renderXILists('t1');
    renderXILists('t2');
}

function addPlayerXI(tid, rosterIndex) {
    const team = state.teams[tid === 't1' ? 0 : 1];
    if (team.xiLocked) return;
    if (team.xi.length >= 11) return showModal("Limit Reached", "You can only select exactly 11 players for your match squad.");

    const p = team.roster.splice(rosterIndex, 1)[0];
    team.xi.push(p);
    renderXILists(tid);
}

function removePlayerXI(tid, xiIndex) {
    const team = state.teams[tid === 't1' ? 0 : 1];
    if (team.xiLocked) return;

    const p = team.xi.splice(xiIndex, 1)[0];
    team.roster.push(p);
    renderXILists(tid);
}

function movePlayerXI(tid, xiIndex, direction) {
    const team = state.teams[tid === 't1' ? 0 : 1];
    if (team.xiLocked) return;
    const newIdx = xiIndex + direction;
    if (newIdx < 0 || newIdx >= team.xi.length) return;

    const temp = team.xi[xiIndex];
    team.xi[xiIndex] = team.xi[newIdx];
    team.xi[newIdx] = temp;
    renderXILists(tid);
}

function renderXILists(tid) {
    const team = state.teams[tid === 't1' ? 0 : 1];
    const benchDiv = document.getElementById(`${tid}-bench`);
    const xiDiv = document.getElementById(`${tid}-playing`);
    benchDiv.innerHTML = '';
    xiDiv.innerHTML = '';

    let activeOs = 0;
    let activeWk = 0;

    // Bench Generation
    team.roster.forEach((p, index) => {
        const el = document.createElement('div');
        el.className = 'xi-player';
        el.innerHTML = `
            <div><strong>${p.name}</strong> ${p.is_foreigner ? '✈️' : ''}<br><span class="role">${p.assigned_role || 'Bat'}</span></div>
            <div style="display:flex; align-items:center; gap:0.5rem;">
                <span class="ovr">${p.batting_ovr || 55}/${p.bowling_ovr || 55}</span>
                ${!team.xiLocked ? `<button class="btn-success" style="padding:0.4rem; font-size:0.7rem;" onclick="addPlayerXI('${tid}', ${index})">➕ ADD</button>` : ''}
            </div>
        `;
        benchDiv.appendChild(el);
    });

    // Playing 11 Generation
    team.xi.forEach((p, index) => {
        if (p.is_foreigner) activeOs++;
        if (p.is_keeper || p.assigned_role === 'Wicket Keeper') activeWk++;

        const el = document.createElement('div');
        el.className = 'xi-player';
        el.innerHTML = `
            <div><span style="color:var(--text-muted); font-size:0.7rem; font-weight:bold; margin-right:4px;">#${index + 1}</span> <strong>${p.name}</strong> ${p.is_foreigner ? '✈️' : ''}<br><span class="role">${p.assigned_role || 'Bat'}</span></div>
            <div style="display:flex; align-items:center; gap:0.3rem;">
                <span class="ovr" style="margin-right:0.5rem;">${p.batting_ovr || 55}/${p.bowling_ovr || 55}</span>
                ${!team.xiLocked ? `
                    <button style="padding:0.2rem 0.4rem; background:#334155; color:white; border:none; border-radius:4px; cursor:pointer;" onclick="movePlayerXI('${tid}', ${index}, -1)">⇧</button>
                    <button style="padding:0.2rem 0.4rem; background:#334155; color:white; border:none; border-radius:4px; cursor:pointer;" onclick="movePlayerXI('${tid}', ${index}, 1)">⇩</button>
                    <button class="btn-danger" style="padding:0.3rem 0.5rem; font-size:0.7rem; margin-left:4px;" onclick="removePlayerXI('${tid}', ${index})">❌</button>
                ` : '<span style="color:var(--text-muted); font-weight:bold; padding:0 0.5rem;">LOCKED</span>'}
            </div>
        `;
        xiDiv.appendChild(el);
    });

    // Update Tracking Dashboards
    document.getElementById(`xi-${tid}-count`).textContent = team.xi.length;
    document.getElementById(`xi-${tid}-count`).style.color = (team.xi.length === 11) ? 'var(--success)' : 'var(--warning)';
    document.getElementById(`xi-${tid}-os`).textContent = activeOs;
    document.getElementById(`xi-${tid}-os`).style.color = (activeOs > 4) ? 'var(--danger)' : 'var(--text-light)';
    document.getElementById(`xi-${tid}-wk`).textContent = activeWk;
    document.getElementById(`xi-${tid}-wk`).style.color = (activeWk < 1) ? 'var(--danger)' : 'var(--success)';

    // Update Toggle/Submit Button
    const btn = document.getElementById(`btn-xi-${tid}`);
    if (team.xiLocked) {
        btn.disabled = false;
        btn.textContent = "CANCEL SUBMIT ❌";
        btn.classList.add('btn-danger');
        btn.classList.remove('btn-success');
    } else {
        btn.classList.add('btn-success');
        btn.classList.remove('btn-danger');
        btn.classList.remove('locked');

        if (team.xi.length === 11 && activeOs <= 4 && activeWk >= 1) {
            btn.disabled = false;
            btn.textContent = "SUBMIT " + team.name.toUpperCase() + " XI";
        } else {
            btn.disabled = true;
            if (activeOs > 4) btn.textContent = "LIMIT 4 OVERSEAS";
            else if (activeWk < 1) btn.textContent = "NEED 1 WICKET KEEPER";
            else btn.textContent = "SELECT EXACTLY 11 PLAYERS";
        }
    }
}

function toggleLockXI(tid) {
    const team = state.teams[tid === 't1' ? 0 : 1];
    if (!team.xiLocked) {
        let activeOs = team.xi.filter(p => p.is_foreigner).length;
        let activeWk = team.xi.filter(p => p.is_keeper || p.assigned_role === 'Wicket Keeper').length;
        if (team.xi.length !== 11 || activeOs > 4 || activeWk < 1) return;
        team.xiLocked = true;
    } else {
        team.xiLocked = false;
    }

    renderXILists(tid);

    if (state.teams[0].xiLocked && state.teams[1].xiLocked) {
        showModal("Pitch Ready!", "The Playing XIs are confirmed. Moving to Toss...", true);
        setTimeout(() => {
            closeModal();
            launchOpenersModal(false);
        }, 2000);
    }
}
let matchData = {
    innings: 1,
    target: null,
    bowlerStats: {},
    lastBowler: null,
    usedBatters: []
};

// --- PHASE 3: LIVE MATCH SIMULATION ---
function initMatchHUD() {
    document.getElementById('playing-xi-screen').style.display = 'none';
    document.getElementById('match-screen').style.display = 'flex';

    let battingTeam = state.teams[matchData.innings === 1 ? 0 : 1];
    let bowlingTeam = state.teams[matchData.innings === 1 ? 1 : 0];

    document.getElementById('batting-team-name').textContent = battingTeam.name;
    document.getElementById('bowling-team-name').textContent = bowlingTeam.name;

    if (matchData.target) {
        document.getElementById('match-target').textContent = matchData.target;
    } else {
        document.getElementById('match-target').textContent = "TBD";
    }

    updateBowlerDropdown();

    document.getElementById('striker-name').textContent = battingTeam.xi[0].name;
    document.getElementById('nonstriker-name').textContent = battingTeam.xi[1].name;
}

function updateBowlerDropdown() {
    const select = document.getElementById('bowler-select');
    select.innerHTML = '';
    const bowlingTeam = state.teams[matchData.innings === 1 ? 1 : 0];

    let hasAvailableBowler = false;
    bowlingTeam.xi.forEach(p => {
        const overs = matchData.bowlerStats[p.name] || 0;
        const isLastBowler = matchData.lastBowler === p.name;

        const opt = document.createElement('option');
        opt.value = p.name;

        let label = `${p.name} (OVR: ${p.bowling_ovr || 55}) [${overs}/4]`;
        if (overs >= 4) {
            opt.disabled = true;
            label += ' - MAX QUOTA';
        } else if (isLastBowler) {
            opt.disabled = true;
            label += ' - RESTING (Just Bowled)';
        } else {
            hasAvailableBowler = true;
        }

        opt.textContent = label;
        select.appendChild(opt);
    });

    if (!hasAvailableBowler) {
        // Failsafe if team mismanagement happens
        bowlingTeam.xi.forEach(p => { select.options[0].disabled = false; });
    }
}

function updateIntentUI(type) {
    const val = parseInt(document.getElementById(`${type}-intent-slider`).value);
    document.getElementById(`${type}-intent-val`).textContent = `${val}%`;
}

let matchStateTracker = {
    t1Ready: false,
    t2Ready: false,
    isProcessing: false
};

function matchReady(tid) {
    if (matchStateTracker.isProcessing) return;

    const btn = document.getElementById(`btn-m-${tid}-ready`);
    btn.classList.add('active');
    btn.classList.remove('btn-warning');
    btn.classList.add('btn-success');
    btn.textContent = "LOCKED IN ✔️";

    if (tid === 't1') matchStateTracker.t1Ready = true;
    if (tid === 't2') matchStateTracker.t2Ready = true;

    if (matchStateTracker.t1Ready && matchStateTracker.t2Ready) {
        playOver();
    }
}

function resetMatchReady() {
    matchStateTracker.t1Ready = false;
    matchStateTracker.t2Ready = false;

    const b1 = document.getElementById('btn-m-t1-ready');
    b1.classList.remove('btn-success', 'active');
    b1.classList.add('btn-warning');
    b1.textContent = "T1 READY";
    b1.disabled = false;

    const b2 = document.getElementById('btn-m-t2-ready');
    b2.classList.remove('btn-success', 'active');
    b2.classList.add('btn-warning');
    b2.textContent = "T2 READY";
    b2.disabled = false;
}

async function playOver(isResume = false) {
    if (!isResume) {
        matchStateTracker.isProcessing = true;
        document.getElementById('btn-m-t1-ready').disabled = true;
        document.getElementById('btn-m-t2-ready').disabled = true;
    }

    const strikerIntent = parseInt(document.getElementById('striker-intent-slider').value);
    const nonStrikerIntent = parseInt(document.getElementById('nonstriker-intent-slider').value);
    const bowlIntent = parseInt(document.getElementById('bowl-intent-slider').value);
    const bowlerName = document.getElementById('bowler-select').value;

    try {
        const res = await fetch('/api/play_over', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                striker_intent: strikerIntent,
                non_striker_intent: nonStrikerIntent,
                bowl_intent: bowlIntent,
                bowler_name: bowlerName
            })
        });
        const data = await res.json();

        if (data.status === 'success') {
            if (!isResume) {
                matchData.lastBowler = bowlerName;
                matchData.bowlerStats[bowlerName] = (matchData.bowlerStats[bowlerName] || 0) + 1;
            }
            await simulateOverVisuals(data.over_summary, data.match_state);
        } else if (data.status === 'all_out') {
            triggerInningsOver(data.match_state);
        } else {
            showModal("Engine Error", data.message);
            matchStateTracker.isProcessing = false;
            resetMatchReady();
        }
    } catch (err) {
        showModal("Connection Error", "JS Crash: " + err.message + "<br><small>" + err.stack + "</small>");
        matchStateTracker.isProcessing = false;
        resetMatchReady();
    }
}

async function simulateOverVisuals(summary, mState) {
    const box = document.getElementById('commentary-box');

    const preLine = document.createElement('div');
    preLine.className = 'comm-line';
    preLine.style.color = 'var(--warning)';
    preLine.textContent = `🎯 ${summary.bowler_name} runs in to bowl the over...`;
    box.appendChild(preLine);

    // Simulate ball-by-ball pacing
    for (const ballComm of summary.commentary) {
        await new Promise(r => setTimeout(r, 400));

        let commClass = "comm-line";
        if (ballComm.includes("OUT!")) commClass += " comm-wicket";
        if (ballComm.includes("4 Run") || ballComm.includes("6 Run")) commClass += " comm-boundary";

        const newLine = document.createElement('div');
        newLine.className = commClass;
        newLine.textContent = ballComm;

        box.appendChild(newLine);
        box.scrollTop = box.scrollHeight;
    }

    // Update live Dashboard after over plays out
    document.getElementById('match-runs').textContent = mState.runs;
    document.getElementById('match-wickets').textContent = mState.wickets;
    document.getElementById('match-overs').textContent = `${Math.floor(mState.balls / 6)}.${mState.balls % 6}`;
    document.getElementById('match-extras').textContent = mState.extras;

    document.getElementById('striker-name').textContent = (summary.wickets_fallen > 0 && !mState.is_all_out) ? "Pending Next Batsman..." : (mState.striker || "ALL OUT");
    document.getElementById('nonstriker-name').textContent = mState.non_striker || "Empty End";

    await new Promise(r => setTimeout(r, 600));

    // Endgame preemptive check
    const isEndgame = mState.is_all_out || (Math.floor(mState.balls / 6) >= 20) || (matchData.innings === 2 && mState.runs >= matchData.target);

    // Determine if Wicket fell strictly before an Endgame finish
    if (summary.wickets_fallen > 0 && !isEndgame) {
        matchData.pendingState = mState;
        matchData.pendingSummary = summary;

        const partialLine = document.createElement('div');
        partialLine.className = 'comm-line';
        partialLine.style.color = '#94a3b8';
        partialLine.textContent = `[OVER PAUSED] Wicket! Select next incoming batsman.`;
        box.appendChild(partialLine);
        box.scrollTop = box.scrollHeight;

        showWicketModal();
    } else {
        finishOverPostProcessor(mState, summary, isEndgame);
    }
}

function finishOverPostProcessor(mState, summary, isEndgame) {
    const box = document.getElementById('commentary-box');
    const sumLine = document.createElement('div');
    sumLine.className = 'comm-line';
    sumLine.style.color = '#94a3b8';
    sumLine.textContent = `[End of Phase] Runs: ${summary.runs_scored} | Wickets: ${summary.wickets_fallen} | Score: ${mState.runs}/${mState.wickets}`;
    box.appendChild(sumLine);
    box.scrollTop = box.scrollHeight;

    if (isEndgame) {
        triggerInningsOver(mState);
    } else {
        matchStateTracker.isProcessing = false;
        updateBowlerDropdown();
        resetMatchReady();
    }
}

function showWicketModal() {
    const select = document.getElementById('next-batter-select');
    select.innerHTML = '';
    const battingTeam = state.teams[matchData.innings === 1 ? 0 : 1];

    battingTeam.xi.forEach(p => {
        if (!matchData.usedBatters.includes(p.name)) {
            const opt = document.createElement('option');
            opt.value = p.name;
            opt.textContent = `${p.name} (OVR: ${p.batting_ovr || 55})`;
            select.appendChild(opt);
        }
    });

    document.getElementById('wicket-modal').style.display = 'flex';
}

async function submitNextBatter() {
    const selected = document.getElementById('next-batter-select').value;
    document.getElementById('wicket-modal').style.display = 'none';

    await fetch('/api/set_next_batter', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ batter_name: selected })
    });
    matchData.usedBatters.push(selected);

    // Check if the wicket perfectly ended the over (ball 6)
    let remainder = matchData.pendingState.balls % 6;
    if (remainder !== 0) {
        // Auto-resume the rest of the partial over
        playOver(true);
    } else {
        // Wicket fell exactly on Ball 6. The over is cleanly finished.
        // Hand off to standard End of over logic!
        finishOverPostProcessor(matchData.pendingState, matchData.pendingSummary);
    }
}

async function triggerInningsOver(mState) {
    if (matchData.innings === 1) {
        matchData.innings = 2;
        matchData.target = mState.runs + 1;
        matchData.usedBatters = [];
        matchData.bowlerStats = {};
        matchData.lastBowler = null;

        document.getElementById('btn-m-t1-ready').style.display = 'none';
        document.getElementById('btn-m-t2-ready').style.display = 'none';
        document.getElementById('match-over-alert').style.display = 'block';

        const box = document.getElementById('commentary-box');
        const overLine = document.createElement('div');
        overLine.className = 'comm-line comm-wicket';
        overLine.textContent = `🛑 INNINGS 1 COMPLETE!`;
        box.appendChild(overLine);
        box.scrollTop = box.scrollHeight;

        showModal("Innings Over", `Target set to ${matchData.target}. Time for the final run chase!`);
        setTimeout(() => {
            closeModal();
            launchOpenersModal(true);
        }, 1500);
    } else {
        // Match physically over
        document.getElementById('btn-m-t1-ready').style.display = 'none';
        document.getElementById('btn-m-t2-ready').style.display = 'none';

        const battingTeam = state.teams[1].name;
        const bowlingTeam = state.teams[0].name;

        let resultMsg = "";
        if (mState.runs >= matchData.target) resultMsg = `${battingTeam} WINS by ${10 - mState.wickets} wickets!`;
        else if (mState.runs === matchData.target - 1) resultMsg = "Match Tied (Super Over required)!";
        else resultMsg = `${bowlingTeam} WINS by ${matchData.target - mState.runs - 1} runs!`;

        showModal('MATCH FINISHED 🏆', resultMsg);
    }
}

async function executeStartSecondInnings() {
    await fetch('/api/start_second_innings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target: matchData.target })
    });

    // Physical Reset for Innings 2 
    document.getElementById('commentary-box').innerHTML = '<div class="comm-line" style="color:white;">☀️ 2nd Innings Begins. Openers walk out to chase the target!</div>';

    initMatchHUD();
    resetMatchReady();

    document.getElementById('match-runs').textContent = 0;
    document.getElementById('match-wickets').textContent = 0;
    document.getElementById('match-overs').textContent = "0.0";
    document.getElementById('match-extras').textContent = 0;

    document.getElementById('btn-m-t1-ready').style.display = 'inline-block';
    document.getElementById('btn-m-t2-ready').style.display = 'inline-block';
    document.getElementById('match-over-alert').style.display = 'none';
    matchStateTracker.isProcessing = false;
}
