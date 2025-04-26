#!/bin/bash

# Create logs for troubleshooting
LOGDIR=/home/LogFiles
mkdir -p $LOGDIR
LOGFILE=$LOGDIR/startup_$(date +"%Y%m%d_%H%M%S").log
exec > >(tee -a "$LOGFILE") 2>&1

echo "[$(date)] Starting deployment setup..."

echo "[$(date)] Installing system dependencies..."
export DEBIAN_FRONTEND=noninteractive
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
    git \
    python3-dev

echo "[$(date)] Checking if dlib is installed..."
if python3 -c "import dlib" 2>/dev/null; then
    echo "[$(date)] dlib is already installed"
else
    echo "[$(date)] Installing dlib..."
    cd /home/site/wwwroot
    # Try multiple methods to install dlib
    pip install dlib --no-cache-dir --verbose || \
    python3 install-dlib.py || \
    echo "[$(date)] WARNING: Failed to install dlib"
fi

echo "[$(date)] Starting application server..."
cd /home/site/wwwroot
gunicorn --bind=0.0.0.0 --timeout 120 app:app