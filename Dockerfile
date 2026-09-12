FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY sql ./sql

# Sin CMD por defecto: docker-compose.yml decide si este contenedor
# corre la API (uvicorn) o el worker de eventos (listener). Un único
# Dockerfile para ambos evita mantener dos imágenes casi idénticas.
