FROM python:3.11-slim

# HF Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the pinned requirements file first
COPY requirements.txt .

# Install packages exactly as defined in requirements.txt (including versions)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Generate the dataset at build time
RUN python create_dataset.py

# Switch to non-root user (required by HF Spaces)
USER 1000

# Declare expected environment variables (values come from HF Space Secrets)
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
