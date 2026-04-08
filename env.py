import numpy as np
import gymnasium as gym
from gymnasium import spaces
from data_loader import load_dataset

# Actions the agent can take
PASS       = 0  # do nothing, advance window
FILTER_IP  = 1  # block source IP in window
CORRELATE  = 2  # look for pattern across last 5 logs
PIVOT_USER = 3  # flag user account
ALERT      = 4  # raise a security alert (costs budget)
QUARANTINE = 5  # full quarantine (high cost, high reward if TP)

WINDOW_SIZE   = 10   # logs visible per step
ALERT_BUDGET  = 20   # max alerts per episode
MAX_STEPS     = 200

class LogHuntEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path: str, curriculum: str = "easy"):
        super().__init__()
        self.features, self.labels = load_dataset(data_path)
        self.curriculum = curriculum  # "easy" | "medium" | "hard"
        self.n_features = self.features.shape[1]

        # Observation: WINDOW_SIZE log rows + [alert_budget_norm, step_norm]
        obs_dim = self.n_features * WINDOW_SIZE + 2
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(6)

        self._ptr = 0
        self._step = 0
        self._alert_budget = ALERT_BUDGET
        self._window = []
        self._window_labels = []
        self._episode_stats = {}

    # ── OpenEnv API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        start = self.np_random.integers(0, len(self.features) - MAX_STEPS - WINDOW_SIZE)
        self._ptr = int(start)
        self._step = 0
        self._alert_budget = ALERT_BUDGET
        self._episode_stats = {"tp": 0, "fp": 0, "correlations": 0, "kill_chains": 0}

        # curriculum: easy = mostly one attack type
        if self.curriculum == "easy":
            # find a slice dominated by one label
            for _ in range(50):
                s = self.np_random.integers(0, len(self.features) - MAX_STEPS)
                chunk_labels = self.labels[s: s + MAX_STEPS]
                unique, counts = np.unique(chunk_labels, return_counts=True)
                if counts.max() / counts.sum() > 0.7:
                    self._ptr = int(s)
                    break

        self._load_window()
        return self.state(), {}

    def step(self, action: int):
        reward = -0.1  # small step cost to discourage passivity
        done = False
        info = {}

        current_labels = self._window_labels
        is_attack = any(l != 0 for l in current_labels)
        attack_types = set(l for l in current_labels if l != 0)

        if action == PASS:
            reward -= 0.2 if is_attack else 0.0  # mild miss penalty

        elif action == FILTER_IP:
            if is_attack:
                reward += 2.0
                self._episode_stats["correlations"] += 1
            else:
                reward -= 1.0  # FP

        elif action == CORRELATE:
            # check last 5 log labels for pattern
            recent = self._window_labels[-5:]
            if len(set(l for l in recent if l != 0)) >= 1:
                reward += 1.0
                self._episode_stats["correlations"] += 1
            else:
                reward += 0.1  # no harm, just slow

        elif action == PIVOT_USER:
            if is_attack and len(attack_types) > 1:
                reward += 3.0  # lateral movement detected
            elif is_attack:
                reward += 1.5
            else:
                reward -= 2.0  # FP is bad

        elif action == ALERT:
            if self._alert_budget <= 0:
                reward -= 3.0  # budget exhausted
            elif is_attack:
                reward += 10.0
                self._alert_budget -= 1
                self._episode_stats["tp"] += 1
            else:
                reward -= 5.0  # false positive
                self._alert_budget -= 1
                self._episode_stats["fp"] += 1

        elif action == QUARANTINE:
            if self._alert_budget <= 0:
                reward -= 5.0
            elif is_attack and len(attack_types) >= 2:
                reward += 20.0  # kill-chain caught
                self._alert_budget -= 2
                self._episode_stats["kill_chains"] += 1
            elif is_attack:
                reward += 8.0
                self._alert_budget -= 1
                self._episode_stats["tp"] += 1
            else:
                reward -= 8.0  # very bad FP
                self._alert_budget -= 2
                self._episode_stats["fp"] += 1

        # advance window
        self._ptr += 1
        self._step += 1
        self._load_window()

        if self._step >= MAX_STEPS or self._alert_budget <= 0:
            done = True
            info = self._episode_stats.copy()
            # bonus for high precision
            tp = self._episode_stats["tp"]
            fp = self._episode_stats["fp"]
            if tp + fp > 0:
                precision = tp / (tp + fp)
                reward += precision * 15.0

        return self.state(), float(reward), done, False, info

    def state(self) -> np.ndarray:
        window_flat = np.array(self._window, dtype=np.float32).flatten()
        meta = np.array([
            self._alert_budget / ALERT_BUDGET,  # budget remaining (0-1)
            self._step / MAX_STEPS,             # episode progress (0-1)
        ], dtype=np.float32)
        return np.concatenate([window_flat, meta])

    def render(self):
        tp = self._episode_stats.get("tp", 0)
        fp = self._episode_stats.get("fp", 0)
        print(f"Step {self._step:3d} | Budget: {self._alert_budget:2d} | "
              f"TP: {tp} FP: {fp} | Attacks in window: {sum(1 for l in self._window_labels if l != 0)}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load_window(self):
        end = min(self._ptr + WINDOW_SIZE, len(self.features))
        start = max(0, end - WINDOW_SIZE)
        self._window = self.features[start:end].tolist()
        self._window_labels = self.labels[start:end].tolist()
        # pad if near end of dataset
        while len(self._window) < WINDOW_SIZE:
            self._window.insert(0, [0.0] * self.n_features)
            self._window_labels.insert(0, 0)
