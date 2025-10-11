# Dockerfile
# -----------

# Stage 1 – build environment
FROM python:3.11-slim AS builder
WORKDIR /app

# install minimal build tools
RUN apt-get update && apt-get install -y --no-install-recommends gcc curl && rm -rf /var/lib/apt/lists/*

# copy requirement list and install to temp layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything
COPY . .

# expose API port
EXPOSE 8000

# healthcheck for docker compose
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1

# default start cmd
CMD ["python", "src/app.py"]