#!/bin/bash
#SBATCH --job-name=AIGC
#SBATCH --nodes=1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=48:00:00                 # Max 48 hours (adjust as needed)
#SBATCH --partition=coursework                 # or: gpu-a100, gpu-rtx, etc. (check your cluster)
#SBATCH --gres=gpu:1                    # 1 GPU (change to 2,4,8 if doing DDP)
#SBATCH --cpus-per-task=12              # Good for data loading
#SBATCH --mem=64G                       # 64 GB RAM (increase if OOM)
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_90
#SBATCH --mail-user=your_email@your_university.edu   # Optional

# ==================== Directory Setup ====================
cd $HOME/CS4487-course-project

# Create log directory if not exists
mkdir -p logs

# ==================== Environment ====================
source venv/bin/activate          # Activate your Python 3.9 venv
export PYTHONPATH=$PWD:$PYTHONPATH
pip install -r requirements.txt   # Install requirements

# ==================== Training Command ====================

# Stage 1: Train backbones (36 epochs)
python train.py

echo "Job finished at $(date)"