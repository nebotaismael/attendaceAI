#!/bin/bash
echo "Installing system dependencies..."
apt-get update
apt-get install -y --no-install-recommends \
    cmake \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gfortran \
    python3-dev

echo "Starting application server..."
gunicorn --bind=0.0.0.0 --timeout 120 app:app