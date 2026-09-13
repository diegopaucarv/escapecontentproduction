FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# torch se instala aparte: el wheel depende del hardware (CPU vs CUDA).
# Default: CPU (~200MB). Para GPU, pasar build args, p.ej.:
#   docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
#                        --build-arg TORCH_PACKAGE=torch==2.12.0+cu126
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_PACKAGE=torch==2.12.0+cpu
RUN pip install --no-cache-dir --extra-index-url ${TORCH_INDEX_URL} ${TORCH_PACKAGE}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY sql ./sql
COPY alembic ./alembic
COPY alembic.ini .

# Sin CMD por defecto: docker-compose.yml decide si este contenedor
# corre la API (uvicorn) o el worker de eventos (listener). Un único
# Dockerfile para ambos evita mantener dos imágenes casi idénticas.
