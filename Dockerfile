# Dockerfile
# Multi-purpose image for semantic_book_recommender (CPU)
# Notes:
# - Builds FAISS (via pip faiss-cpu) and sentence-transformers.
# - The build can be heavy because torch + transformer models are downloaded at pip-time or first run.
# - If you prefer smaller image and faster builds, use a dedicated PyTorch CPU base image and/or pull prebuilt wheels.

FROM python:3.10-slim

# metadata
LABEL maintainer="you@example.com"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# set workdir
WORKDIR /app

# install system dependencies for building some packages and for faiss
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       wget \
       curl \
       ca-certificates \
       pkg-config \
       libopenblas-dev \
       libomp-dev \
       libfftw3-dev \
       libblas-dev \
       liblapack-dev \
       gfortran \
       libstdc++6 \
       libssl-dev \
       libffi-dev \
       python3-dev \
    && rm -rf /var/lib/apt/lists/*

# copy requirements and project
COPY requirements.txt /app/requirements.txt
COPY . /app

# Upgrade pip and install python deps
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip --no-cache-dir install -r /app/requirements.txt

# make entrypoint executable
COPY docker/entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# expose streamlit default port
EXPOSE 8501

# Use a non-root user for runtime (optional)
RUN useradd --create-home appuser || true
USER appuser
WORKDIR /home/appuser
COPY --chown=appuser:appuser . /home/appuser/app
WORKDIR /home/appuser/app

# Entrypoint: build index if missing, then run streamlit
ENTRYPOINT ["/home/appuser/app/docker-entrypoint.sh"]
