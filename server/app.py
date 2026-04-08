from fastapi import FastAPI
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from env import LogHuntEnv
import numpy as np

app = FastAPI()
env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum="easy")
obs, _ = env.reset()

@app.get("/")
def home():
    return {"status": "LogHuntEnv is running!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/tasks")
def tasks():
    return {"tasks": [
        {"id": "easy", "difficulty": "easy"},
        {"id": "medium", "difficulty": "medium"},
        {"id": "hard", "difficulty": "hard"}
    ]}

@app.post("/reset")
def reset(task_id: str = "easy"):
    global obs, env
    env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum=task_id)
    obs, _ = env.reset()
    return {"observation": obs.tolist()}

@app.post("/step")
def step(action: int = 0):
    global obs
    obs, reward, done, _, info = env.step(action)
    return {
        "observation": obs.tolist(),
        "reward": float(reward),
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return {"observation": obs.tolist()}