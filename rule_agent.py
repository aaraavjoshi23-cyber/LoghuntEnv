import numpy as np

class RuleBasedAgent:
    """Heuristic agent — good baseline for showing improvement."""

    def __init__(self, n_features, window_size=10):
        self.n_features = n_features
        self.ws = window_size

    def act(self, obs):
        # last row of window = most recent log
        window = obs[:self.n_features * self.ws].reshape(self.ws, self.n_features)
        recent = window[-1]

        flow_bytes  = recent[7]   # 'Flow Bytes/s' index
        syn_flags   = recent[13]  # 'SYN Flag Count'
        rst_flags   = recent[14]  # 'RST Flag Count'
        budget_norm = obs[-2]

        if budget_norm < 0.1:
            return 0  # PASS — out of budget

        if syn_flags > 2.0:
            return 4  # ALERT — likely SYN flood / port scan

        if flow_bytes > 3.0 and rst_flags > 1.5:
            return 3  # PIVOT_USER — lateral movement pattern

        if flow_bytes > 2.0:
            return 2  # CORRELATE

        return 0  # PASS