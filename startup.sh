#!/bin/bash
echo "[$(date)] Starting deployment setup..."

# Create a log file
LOGFILE=/home/site/wwwroot/startup_log.txt
exec > >(tee -a "$LOGFILE") 2>&1

echo "[$(date)] Checking if system dependencies are installed..."
# Check if cmake is installed
if ! command -v cmake &> /dev/null; then
    echo "Installing system dependencies..."
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
      python3-dev
else
    echo "CMake is already installed, skipping system dependencies installation"
fi

echo "[$(date)] Verifying Python packages..."
# Check if dlib is installed
python3 -c "import dlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dlib..."
    pip install dlib --no-cache-dir
fi

echo "[$(date)] Starting Gunicorn server..."
gunicorn --bind=0.0.0.0 --timeout 120 app:app