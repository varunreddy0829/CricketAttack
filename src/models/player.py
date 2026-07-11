# src/models/player.py

class Batter:
    def __init__(self, name: str, ovr: int, career_runs: int, career_balls: int, fours: int, sixes: int, dismissals: int, intent: int = 50):
        self.name = name
        self.ovr = ovr
        self.career_runs = career_runs
        self.career_balls = career_balls
        self.fours = fours
        self.sixes = sixes
        self.dismissals = dismissals
        self.intent = intent
        
        # Calculate derived stats, handling division by zero
        self.avg = self.career_runs / self.dismissals if self.dismissals > 0 else self.career_runs
        self.sr = (self.career_runs / self.career_balls) * 100.0 if self.career_balls > 0 else 0.0

    def __repr__(self):
        return f"Batter({self.name}, OVR={self.ovr}, Avg={self.avg:.2f}, SR={self.sr:.2f}, Intent={self.intent})"


class Bowler:
    def __init__(self, name: str, ovr: int, eco: float, wkt: int, intent: int = 50, legal_balls: int = 0, style: str = "Pace"):
        self.name = name
        self.ovr = ovr
        self.eco = eco
        self.wkt = wkt
        self.intent = intent
        # Career legal balls bowled. Needed by the Stage-3 ghost-stat wicket
        # factor to derive a smoothed bowling strike rate (balls per wicket).
        self.legal_balls = legal_balls
        # "Pace" or "Spin" — drives the pitch-conditions stage (dusty pitches
        # favour spinners, green ones favour pacers).
        self.style = style

    def __repr__(self):
        return f"Bowler({self.name}, OVR={self.ovr}, Eco={self.eco}, Wickets={self.wkt}, Balls={self.legal_balls}, {self.style}, Intent={self.intent})"
