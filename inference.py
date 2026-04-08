import sys
import os
import numpy as np

from openai import OpenAI

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ["API_KEY"]
)

# =========================
# DATASET CREATION
# =========================
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

create_dataset()

# =========================
# IMPORT ENV
# =========================
from env import LogHuntEnv

TASKS = ["easy", "medium", "hard"]

# =========================
# AI AGENT
# =========================
def agent_decision(obs):
    try:
        prompt = f"""
        You are a cybersecurity AI.
        Classify the network activity into one of these classes:
        0 = BENIGN
        1 = DDoS
        2 = PortScan
        3 = Bot
        4 = Suspicious
        5 = Unknown
        Data:
        {str(obs)[:300]}
        Only return a number (0-5).
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a cybersecurity expert."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

        action = int(answer) if answer.isdigit() else 0
        return max(0, min(action, 5))

    except Exception:
        return 0


# =========================
# RUN TASK
# =========================
def run_task(task_id):
    try:
        env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum=task_id)
        obs, _ = env.reset()

        print(f"[START] task={task_id}", flush=True)

        total_reward = 0
        steps = 0

        for _ in range(50):  # reduced for API safety
            action = agent_decision(obs)

            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            steps += 1

            print(f"[STEP] step={steps} action={action} reward={round(float(reward), 4)} done={done}", flush=True)

            if done:
                break

        score = float(min(max(total_reward / 300, 0.01), 0.99))

        print(f"[END] task={task_id} score={round(score, 4)} steps={steps}", flush=True)

        return score

    except Exception as e:
        print(f"[START] task={task_id}", flush=True)
        print(f"[STEP] step=1 action=0 reward=0.0 done=True", flush=True)
        print(f"[END] task={task_id} score=0.0 steps=1", flush=True)
        return 0.0


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
