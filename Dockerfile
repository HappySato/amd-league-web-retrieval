# CPU image for the retrieval app. The model stays in a separate vLLM process
# (see scripts/amd_vllm_notebook.py). Build:
#   docker build -t iwir .
# Run, with questions mounted in:
#   docker run --rm -v "$PWD/input:/app/input:ro" -v "$PWD/output:/app/output" iwir
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OUTPUT_DIR=/app/output \
    QUESTION_TIMEOUT_SECONDS=24 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY iwir ./iwir
COPY input ./input
COPY scripts ./scripts
COPY docs ./docs
COPY README.md LICENSE ./

RUN mkdir -p /app/output /app/input

# The harness starts the container with this command. CMD (not ENTRYPOINT)
# lets `docker run IMAGE python3 /app/app.py ...` replace it cleanly.
CMD ["python3", "/app/app.py"]
