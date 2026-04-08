# LogHuntEnv 🔍

**A real-world AI cybersecurity environment for network intrusion detection.**

An AI agent analyzes live network traffic logs and must detect attacks (DDoS, PortScan, Bot activity) while managing a limited alert budget — just like a real Security Operations Center (SOC) analyst.

---

## 🌐 HF Space

> Deploy URL will be listed here after deployment.

---

## 📋 Environment Description

| Property | Value |
|---|---|
| Task type | Network intrusion detection (cybersecurity) |
| Based on | CICIDS2017 network traffic dataset patterns |
| Attack types | DDoS, PortScan, Bot, Benign |
| Episode length | Up to 200 steps |
| Alert budget | 20 alerts per episode |

---

## 👁️ Observation Space

- **Type:** `Box(float32)`
- **Shape:** `(202,)`
- **Description:**
  - Indices `0–199`: Flattened window of **10 log rows × 20 network features** (packet counts, flow duration, flag counts, byte rates, etc.)
  - Index `200`: Alert budget remaining, normalized `0.0–1.0`
  - Index `201`: Episode progress, normalized `0.0–1.0`

---

## 🎮 Action Space

- **Type:** `Discrete(6)`

| Action ID | Name | Description |
|---|---|---|
| `0` | PASS | Do nothing, move to next log window |
| `1` | FILTER_IP | Block the suspicious source IP |
| `2` | CORRELATE | Investigate attack pattern across last 5 logs |
| `3` | PIVOT_USER | Flag the user account as suspicious |
| `4` | ALERT | Raise a security alert (costs 1 budget) |
| `5` | QUARANTINE | Full system quarantine (costs 2 budget) |

---

## 🏆 Tasks

| Task | Description | Score Threshold |
|---|---|---|
| `easy` | Single dominant attack type | 0.3 |
| `medium` | Multiple simultaneous attack types | 0.5 |
| `hard` | Full kill-chains across attack stages | 0.7 |

---

## 💰 Reward Function

| Action | Outcome | Reward |
|---|---|---|
| ALERT | True positive (real attack) | +10 |
| ALERT | False positive (benign traffic) | -5 |
| QUARANTINE | Kill-chain caught (2+ attack types) | +20 |
| QUARANTINE | Single attack caught | +8 |
| QUARANTINE | False positive | -8 |
| FILTER_IP | Attack window | +2 |
| FILTER_IP | Benign traffic | -1 |
| PIVOT_USER | Lateral movement detected | +3 |
| CORRELATE | Pattern found | +1 |
| PASS | During attack window | -0.2 |
| Episode bonus | High precision (TP / (TP+FP)) × 15 | up to +15 |

**Score = clamp(total_reward / 300, 0.0, 1.0)**

---

## ⚙️ Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
Create a `.env` file in the project root:
```
API_BASE_URL=https://api-inference.huggingface.co/v1
MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
HF_TOKEN=hf_... (your token)
```

### 3. Generate dataset
```bash
python create_dataset.py
```

### 4. Run the API server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### 5. Run inference
```bash
python inference.py
```

---

## 🔌 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks |
| `POST` | `/reset` | Reset environment `{"task_id": "easy"}` |
| `POST` | `/step` | Take a step `{"action": 0}` |
| `GET` | `/state` | Get current observation |

---

## 🐳 Docker

```bash
# Build
docker build -t loghuntenv .

# Run
docker run -p 7860:7860 \
  -e API_BASE_URL=https://api-inference.huggingface.co/v1 \
  -e MODEL_NAME=HuggingFaceH4/zephyr-7b-beta \
  -e HF_TOKEN=your_token \
  loghuntenv
```

---

## ✅ Pre-submission Validation

```bash
python validate.py
```

---

## 📁 Project Structure

```
loghuntenv/
├── inference.py       # Baseline inference script (LLM agent)
├── app.py             # FastAPI server (OpenEnv spec)
├── env.py             # Gymnasium environment
├── data_loader.py     # Dataset loading & preprocessing
├── create_dataset.py  # Synthetic dataset generator
├── openenv.yaml       # OpenEnv specification file
├── Dockerfile         # Container definition
├── requirements.txt   # Python dependencies
├── validate.py        # Pre-submission validator
└── .env               # Environment variables (not committed)
```
