# Backend Dockerfile
FROM python:3.11-slim AS backend

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (for scientific stack wheels; keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps first for better caching
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy backend code
COPY backend/ /app/

# Expose FastAPI port
EXPOSE 8001

# Healthcheck optional (ping the health route)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/api/').read()" || exit 1

# Run
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
