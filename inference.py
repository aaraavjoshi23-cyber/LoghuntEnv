import json
import os
import time
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from env import LogHuntEnv

# Load environment variables from .env file
load_dotenv()

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME   = os.environ.get("MODEL_NAME")
HF_TOKEN     = os.environ.get("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

TASKS = ["easy", "medium", "hard"]

def ask_llm(obs, step_num, task_id):
    # obs is 202 values: (10 logs * 20 features) + budget + progress
    logs = obs[:200].reshape(10, 20)
    features_mean = logs.mean(axis=0).round(3).tolist() # Average of each feature over the window
    budget = round(obs[200], 2)
    progress = round(obs[201], 2)

    prompt = f"""You are a cybersecurity AI agent analyzing network traffic logs.
Task difficulty: {task_id}  |  Step: {step_num}
Alert Budget Left: {budget} | Progress: {progress}
Window Summary (Average of 20 features over last 10 logs):
{features_mean}

Your job is to detect attacks (DDoS, PortScan, Bot). Choose ONE action (0-5):
0 = PASS        (do nothing, advance)
1 = FILTER_IP   (block source IP)
2 = CORRELATE   (check patterns)
3 = PIVOT_USER  (flag lateral movement)
4 = ALERT       (raise security alert - costs budget)
5 = QUARANTINE  (full system lockdown - high cost/reward)

Reply with ONLY a single digit (0-5):"""

    for attempt in range(3):  # 3 Retries for rate limits
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.1,
            )
            reply = response.choices[0].message.content.strip()
            for char in reply:
                if char in "012345":
                    return int(char)
            break # If we got a response but no digit, stop retrying and use fallback
        except RateLimitError:
            wait_time = (attempt + 1) * 2 # 2, 4, 6 seconds
            time.sleep(wait_time)
        except Exception:
            break # Other errors use immediate fallback

        pass

    # Rule-based fallback if LLM call fails
    window_flat = obs[:-2]
    try:
        features_2d = window_flat.reshape(10, -1)
        syn_mean = features_2d[:, 13].mean()   # SYN Flag Count column
        pkt_mean = features_2d[:, 2].mean()    # Total Fwd Packets column
        if syn_mean > 0.6:
            return 4  # ALERT — high SYN flood signal
        elif syn_mean > 0.4:
            return 2  # CORRELATE — moderate signal
        elif pkt_mean > 0.7:
            return 1  # FILTER_IP — high packet rate
        else:
            return 0  # PASS
    except Exception:
        return 0

def run_task(task_id):
    env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum=task_id)
    obs, _ = env.reset()

    print(json.dumps({
        "type": "[START]",
        "task_id": task_id,
        "observation": obs.tolist()
    }))

    total_reward = 0.0
    steps = 0
    info = {}

    for i in range(200):
        action = ask_llm(obs, i, task_id)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        steps += 1

        # Small delay to respect RPM (Requests Per Minute) limits
        time.sleep(0.2) 

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

    # Normalize score to 0.0–1.0
    score = float(min(max(total_reward / 300.0, 0.0), 1.0))

    print(json.dumps({
        "type": "[END]",
        "task_id": task_id,
        "total_reward": float(total_reward),
        "score": score,
        "steps": steps,
        "info": info
    }))

    return score

if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
