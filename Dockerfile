FROM python:3.11-slim

# HF Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install all packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy rest of files
COPY . .

# Generate dataset at build time
RUN python create_dataset.py

# Switch to non-root user (required by HF Spaces)
USER 1000

# Environment variables from HF Space Secrets
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
