# Multi-stage build — smaller final image, non-root user, data/ as volume.

# ---- Stage 1: build wheel ----
FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY covid_ndd_extraction/ covid_ndd_extraction/
RUN pip install --no-cache-dir build && python -m build --wheel

# ---- Stage 2: runtime ----
FROM python:3.11-slim

# System deps (OpenCV requires libgl; Tesseract optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m appuser

WORKDIR /app

# Copy wheel from builder and install with all extras
COPY --from=builder /build/dist/*.whl ./
RUN pip install --no-cache-dir *.whl[graph,vision]

# Data directory mounted at runtime (not baked into image)
VOLUME ["/app/data"]

USER appuser

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["covid-ndd"]
CMD ["--help"]
