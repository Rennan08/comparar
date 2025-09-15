FROM python:3.11-slim

# Dependências nativas (OpenCV/Scikit-image) + build-essential
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/face-compare
COPY requirements.txt ./requirements.txt
