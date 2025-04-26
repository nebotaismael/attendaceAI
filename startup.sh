#!/bin/bash
# filepath: e:\Files\powerub-master\AI Club\attendance_verification_system\startup.sh

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
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    libglib2.0-0 

echo "[$(date)] Starting application server..."
cd /home/site/wwwroot
gunicorn --bind=0.0.0.0 --timeout 120 application:app