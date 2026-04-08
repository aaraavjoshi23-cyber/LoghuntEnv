import json
import numpy as np
import os
import sys

# ── Create dataset before anything else ──────────────────────────────────────
def create_dataset():
    import pandas as pd
    os.makedirs('data', exist_ok=True)
    if not os.path.exists('data/CICIDS2017_sample.csv'):
        np.random.seed(42)
        n = 5000
        data = {
            ' Label': np.random.choice(['BENIGN','DDoS','PortScan','Bot'], n),
            'Flow Duration': np.random.randint(0, 100000, n),
            'Total Fwd Packets': np.random.randint(1, 100, n),
            'Total Backward Packets': np.random.randint(1, 100, n),
            'Total Length of Fwd Packets': np.random.randint(0, 5000, n),
            'Total Length of Bwd Packets': np.random.randint(0, 5000, n),
            'Fwd Packet Length Max': np.random.randint(0, 1500, n),
            'Bwd Packet Length Max': np.random.randint(0, 1500, n),
            'Flow Bytes/s': np.random.uniform(0, 1000000, n),
            'Flow Packets/s': np.random.uniform(0, 1000, n),
            'Flow IAT Mean': np.random.uniform(0, 100000, n),
            'Fwd IAT Total': np.random.uniform(0, 100000, n),
            'Bwd IAT Total': np.random.uniform(0, 100000, n),
            'Fwd PSH Flags': np.random.randint(0, 2, n),
            'SYN Flag Count': np.random.randint(0, 10, n),
            'RST Flag Count': np.random.randint(0, 5, n),
            'URG Flag Count': np.random.randint(0, 2, n),
            'Packet Length Mean': np.random.uniform(0, 1500, n),
            'Packet Length Std': np.random.uniform(0, 500, n),
            'Average Packet Size': np.random.uniform(0, 1500, n),
            'Avg Fwd Segment Size': np.random.uniform(0, 1500, n),
        }
        pd.DataFrame(data).to_csv('data/CICIDS2017_sample.csv', index=False)
        print("Dataset created!", flush=True)

# Create dataset first before importing env
create_dataset()

# ── Now import env ────────────────────────────────────────────────────────────
from env import LogHuntEnv

TASKS = ["easy", "medium", "hard"]

def run_task(task_id):
    try:
        env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum=task_id)
        obs, _ = env.reset()

        print(json.dumps({
            "type": "[START]",
            "task_id": task_id,
            "observation": obs.tolist()
        }), flush=True)

        total_reward = 0
        steps = 0

        for _ in range(200):
            action = np.random.randint(0, 6)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1

            print(json.dumps({
                "type": "[STEP]",
                "task_id": task_id,
                "step": steps,
                "action": int(action),
                "reward": float(reward),
                "done": done
            }), flush=True)

            if done:
                break

        score = float(min(max(total_reward / 300, 0.0), 1.0))

        print(json.dumps({
            "type": "[END]",
            "task_id": task_id,
            "total_reward": float(total_reward),
            "score": score,
            "steps": steps,
            "info": info
        }), flush=True)

        return score

    except Exception as e:
        print(json.dumps({
            "type": "[END]",
            "task_id": task_id,
            "total_reward": 0.0,
            "score": 0.0,
            "steps": 0,
            "error": str(e)
        }), flush=True)
        return 0.0

if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
