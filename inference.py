import json
import os
import numpy as np
from openai import OpenAI
from env import LogHuntEnv

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.environ.get("OPENAI_API_KEY", "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")

client = OpenAI(base_url=API_BASE_URL, api_key="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")

TASKS = ["easy", "medium", "hard"]

def ask_llm(obs, step_num, task_id):
    obs_summary = obs.tolist()[:10]
    prompt = f"""You are a cybersecurity AI agent analyzing network logs.
Task: {task_id}, Step: {step_num}
Observation (first 10 values): {obs_summary}
Choose ONE action - reply with ONLY a single digit:
0 = PASS, 1 = FILTER_IP, 2 = CORRELATE, 3 = ESCALATE, 4 = ALERT"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.1,
        )
        reply = response.choices[0].message.content.strip()
        for char in reply:
            if char in "01234":
                return int(char)
    except:
        pass
    window_flat = obs[:-2]
    syn_mean = window_flat.reshape(10, -1)[:, 13].mean()
    if syn_mean > 0.5:
        return 4
    elif syn_mean > 0.3:
        return 1
    return 0

def run_task(task_id):
    env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum=task_id)
    obs, _ = env.reset()

    print(json.dumps({"type": "[START]", "task_id": task_id, "observation": obs.tolist()}))

    total_reward = 0
    steps = 0
    info = {}

    for i in range(200):
        action = ask_llm(obs, i, task_id)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1

        print(json.dumps({"type": "[STEP]", "task_id": task_id, "step": steps, "action": int(action), "reward": float(reward), "done": done}))

        if done:
            break

    score = min(max(total_reward / 300, 0.0), 1.0)

    print(json.dumps({"type": "[END]", "task_id": task_id, "total_reward": float(total_reward), "score": float(score), "steps": steps, "info": info}))

    return score

if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
