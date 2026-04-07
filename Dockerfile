FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    gymnasium \
    stable-baselines3 \
    fastapi \
    uvicorn \
    pydantic
RUN python create_dataset.py

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]