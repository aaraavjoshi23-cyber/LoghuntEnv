from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Optional, List
from env import LogHuntEnv
import numpy as np

# Rate limiting (User Limiting)
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="LogHuntEnv", description="AI Cybersecurity Network Log Analysis Environment")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Typed Request / Response Models (OpenEnv spec) ──────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"

class ResetResponse(BaseModel):
    observation: List[float]
    task_id: str

class StepRequest(BaseModel):
    action: int

class StepResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    info: dict

class StateResponse(BaseModel):
    observation: List[float]

class TaskInfo(BaseModel):
    id: str
    description: str
    difficulty: str
    reward_threshold: float

# ── Global state ─────────────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    "easy":   ("Detect single attack type in network logs", "easy",   0.3),
    "medium": ("Detect multiple simultaneous attack types", "medium", 0.5),
    "hard":   ("Detect full kill-chains across attack stages", "hard", 0.7),
}

current_task_id = "easy"
env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum=current_task_id)
obs, _ = env.reset()

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
@limiter.limit("5/minute")
def home(request: Request):
    return {"status": "ok", "env": "LogHuntEnv", "version": "1.0"}

@app.get("/health", summary="Health check")
def health(request: Request):
    return {"status": "ok"}

@app.get("/tasks", summary="List all available tasks", response_model=List[TaskInfo])
@limiter.limit("20/minute")
def list_tasks(request: Request):
    return [
        TaskInfo(id=tid, description=desc, difficulty=diff, reward_threshold=thresh)
        for tid, (desc, diff, thresh) in TASK_DESCRIPTIONS.items()
    ]

@app.post("/reset", summary="Reset the environment", response_model=ResetResponse)
@limiter.limit("5/minute")
def reset(request: Request, body: ResetRequest = ResetRequest()):
    global env, obs, current_task_id
    task_id = body.task_id if body.task_id in TASK_DESCRIPTIONS else "easy"
    current_task_id = task_id
    env = LogHuntEnv("data/CICIDS2017_sample.csv", curriculum=task_id)
    obs, _ = env.reset()
    return ResetResponse(observation=obs.tolist(), task_id=task_id)

@app.post("/step", summary="Take one environment step", response_model=StepResponse)
@limiter.limit("60/minute")
def step(request: Request, body: StepRequest):
    global obs
    obs, reward, done, _, info = env.step(body.action)
    return StepResponse(
        observation=obs.tolist(),
        reward=float(reward),
        done=done,
        info=info
    )

@app.get("/state", summary="Get current observation", response_model=StateResponse)
@limiter.limit("60/minute")
def state(request: Request):
    return StateResponse(observation=obs.tolist())
