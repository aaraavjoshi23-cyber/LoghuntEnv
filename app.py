from fastapi import FastAPI
from env import LogHuntEnv
import numpy as np

app = FastAPI()
env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum="easy")
obs, _ = env.reset()

@app.get("/")
def home():
    return {"status": "LogHuntEnv is running!"}

@app.post("/reset")
def reset():
    global obs
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