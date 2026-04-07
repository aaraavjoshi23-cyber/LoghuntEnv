import json
import numpy as np
from env import LogHuntEnv

TASKS = ["easy", "medium", "hard"]

def run_task(task_id):
    env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum=task_id)
    obs, _ = env.reset()
    
    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "observation": obs.tolist()
    }))
    
    total_reward = 0
    steps = 0
    info = {}
    
    for _ in range(200):
        # Rule-based: if recent window has anomaly signals, alert
        window_flat = obs[:-2]
        syn_mean = window_flat.reshape(10, -1)[:, 13].mean()
        
        if syn_mean > 0.5:
            action = 4  # ALERT
        elif syn_mean > 0.3:
            action = 1  # FILTER_IP
        elif _ % 5 == 0:
            action = 2  # CORRELATE
        else:
            action = 0  # PASS
        
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
        }))
        
        if done:
            break
    
    # normalize score to 0.0-1.0
    score = min(max(total_reward / 300, 0.0), 1.0)
    
    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "total_reward": float(total_reward),
        "score": float(score),
        "steps": steps,
        "info": info
    }))
    
    return score

if __name__ == "__main__":
    for task in TASKS:
        run_task(task)