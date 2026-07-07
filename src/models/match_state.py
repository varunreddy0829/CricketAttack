# src/models/match_state.py

class MatchState:
    def __init__(self, lineup: list):
        """
        lineup: list of Batter objects
        """
        self.lineup = lineup
        self.runs = 0
        self.wickets = 0
        self.balls = 0
        self.extras = 0
        self.striker_index = 0
        self.non_striker_index = 1
        self.next_batter_index = 2
        self.target = None

    def get_striker(self):
        if self.striker_index is not None and self.striker_index < len(self.lineup):
            return self.lineup[self.striker_index]
        return None

    def get_non_striker(self):
        if self.non_striker_index is not None and self.non_striker_index < len(self.lineup):
            return self.lineup[self.non_striker_index]
        return None

    def rotate_strike(self):
        self.striker_index, self.non_striker_index = self.non_striker_index, self.striker_index

    def handle_wicket(self):
        self.wickets += 1
        # DO NOT automatically bring in the next batter.
        # Set striker to None to force frontend manual selection prompt
        self.striker_index = None

    def retire(self, which: str):
        """Permanently retire the striker or non-striker (no return, unlike
        real cricket's 'retired hurt'). Counts as a wicket down for the innings
        but is not credited to any bowler — callers must not touch bowl_card."""
        self.wickets += 1
        if which == "striker":
            self.striker_index = None
        elif which == "non_striker":
            self.non_striker_index = None
        else:
            raise ValueError("which must be 'striker' or 'non_striker'")

    def add_runs(self, runs: int):
        self.runs += runs

    def add_extra(self):
        self.runs += 1
        self.extras += 1

    def add_ball(self):
        self.balls += 1

    def is_all_out(self) -> bool:
        return self.wickets >= min(10, len(self.lineup) - 1)
