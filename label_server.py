import json
import os
import time
import subprocess
from flask import Flask, jsonify, request, render_template_string
import threading

app = Flask(__name__)

HISTORICAL_PATH = 'data/players_historical.json'
LABELS_PATH = 'data/player_labels.json'

try:
    with open(HISTORICAL_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
        data = json.loads(text[text.find('['):])
        all_players = [p['name'] for p in data]
except Exception as e:
    print("Error loading historical data:", e)
    all_players = []

active_locks = {}

def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_labels(labels):
    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump(labels, f, indent=2)

def get_next_player():
    labels = load_labels()
    now = time.time()
    for p in all_players:
        if p not in labels:
            if p not in active_locks or (now - active_locks[p] > 120):
                active_locks[p] = now
                return p
    return None

HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Cricket Player Labeler</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: white; display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; }
        .card { background: #1e293b; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06); text-align: center; width: 90%; max-width: 400px; }
        h1 { margin-bottom: 0.5rem; font-size: 2rem; color: #38bdf8; }
        .progress { color: #94a3b8; font-size: 0.9rem; margin-bottom: 2rem; }
        .question-group { margin-bottom: 2rem; }
        .question-title { font-size: 1.2rem; margin-bottom: 1rem; }
        .buttons { display: flex; justify-content: center; gap: 1rem; }
        button { flex: 1; padding: 1rem; border: none; border-radius: 8px; font-size: 1.1rem; font-weight: bold; cursor: pointer; transition: transform 0.1s, opacity 0.1s; }
        button:active { transform: scale(0.95); }
        .btn-yes { background: #10b981; color: white; }
        .btn-no { background: #ef4444; color: white; }
        .loader { display: none; color: #38bdf8; margin-bottom: 1rem; }
        .done { color: #10b981; font-size: 1.5rem; display: none; }
    </style>
</head>
<body>
    <div class="card" id="card">
        <div class="loader" id="loader">Loading Next Player...</div>
        <h1 id="playerName">Player Name</h1>
        <div class="progress" id="progressText">0 / 0 Labeled</div>
        <div class="question-group">
            <div class="question-title">Is this player a Foreigner (Overseas)?</div>
            <div class="buttons">
                <button class="btn-yes" onclick="answer(true)">YES</button>
                <button class="btn-no" onclick="answer(false)">NO</button>
            </div>
        </div>
    </div>
    <div class="done" id="doneMessage">🎉 All players labeled! Amazing teamwork!</div>

    <script>
        let currentPlayer = null;

        async function fetchNext() {
            document.getElementById("loader").style.display = "block";
            document.getElementById("playerName").style.display = "none";
            
            try {
                const res = await fetch('/api/next');
                const data = await res.json();
                document.getElementById("progressText").innerText = `${data.progress} / ${data.total} Labeled`;
                
                if (data.done) {
                    document.getElementById("card").style.display = "none";
                    document.getElementById("doneMessage").style.display = "block";
                } else {
                    currentPlayer = data.player;
                    document.getElementById("playerName").innerText = currentPlayer;
                    document.getElementById("loader").style.display = "none";
                    document.getElementById("playerName").style.display = "block";
                }
            } catch(e) { console.error(e); }
        }

        async function answer(is_foreigner) {
            if(!currentPlayer) return;
            const payload = {
                name: currentPlayer,
                is_foreigner: is_foreigner
            };
            currentPlayer = null;
            document.getElementById("loader").style.display = "block";
            
            try {
                await fetch('/api/submit', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                fetchNext();
            } catch(e) { console.error(e); }
        }

        fetchNext();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_UI)

@app.route('/api/next', methods=['GET'])
def api_next():
    labels = load_labels()
    p = get_next_player()
    if p is None:
        return jsonify({"done": True, "progress": len(labels), "total": len(all_players)})
    return jsonify({"done": False, "player": p, "progress": len(labels), "total": len(all_players)})

@app.route('/api/submit', methods=['POST'])
def api_submit():
    data = request.json
    name = data.get('name')
    is_f = data.get('is_foreigner')
    
    if name:
        labels = load_labels()
        labels[name] = {"is_foreigner": is_f}
        save_labels(labels)
        if name in active_locks:
            del active_locks[name]
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, debug=False)
