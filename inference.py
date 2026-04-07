import json
import os
import numpy as np
from openai import OpenAI
from env import LogHuntEnv

# Read environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Setup OpenAI client pointing to HuggingFace
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASKS = ["easy", "medium", "hard"]

ACTIONS = {
    0: "PASS - do nothing",
    1: "FILTER_IP - block suspicious IP",
    2: "CORRELATE - correlate log events",
    3: "ESCALATE - escalate to human analyst",
    4: "ALERT - raise security alert"
}

def ask_llm(obs, step_num, task_id):
    """Ask the LLM what action to take given the observation."""
    obs_summary = obs.tolist()[:10]  # Send only first 10 values to keep it short

    prompt = f"""You are a cybersecurity AI agent analyzing network logs.
Task difficulty: {task_id}
Step: {step_num}
Recent network observation (first 10 values): {obs_summary}

Choose ONE action by responding with ONLY a single digit (0-4):
0 = PASS (do nothing, traffic looks normal)
1 = FILTER_IP (block suspicious IP address)
2 = CORRELATE (correlate log events for patterns)
3 = ESCALATE (escalate to human analyst)
4 = ALERT (raise security alert, high threat detected)

Respond with ONLY the digit, nothing else."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.1,
        )
        reply = response.choices[0].message.content.strip()
        # Extract first digit from reply
        for char in reply:
            if char in "01234":
                return int(char)
    except Exception as e:
        pass  # Fall back to rule-based if LLM fails

    # Fallback rule-based action
    window_flat = obs[:-2]
    syn_mean = window_flat.reshape(10, -1)[:, min(13, window_flat.reshape(10, -1).shape[1]-1)].mean()
    if syn_mean > 0.5:
        return 4
    elif syn_mean > 0.3:
        return 1
    return 0


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

    for i in range(200):
        action = ask_llm(obs, i, task_id)

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
