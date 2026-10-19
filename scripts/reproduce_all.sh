#!/usr/bin/env bash
# ==============================================================================
# Master Reproduction Script for Machine Learning Assignment 2
# Roll Number: M26CSE005
# ==============================================================================
set -e

echo "======================================================================"
echo "Starting Full Reproduction Pipeline"
echo "======================================================================"

# 1. Create output directories
mkdir -p logs figures splits

# 2. Run Question 1: Decision Trees and Bagging
echo "[1/4] Running Question 1..."
python3 scripts/run_q1.py

# 3. Run Question 2: SAMME Multi-class Boosting
echo "[2/4] Running Question 2..."
python3 scripts/run_q2.py

# 4. Run Question 3: Bias-Variance Decomposition (R=30 resamples)
echo "[3/4] Running Question 3..."
python3 scripts/run_q3.py

# 5. Generate publication figures
echo "[4/4] Generating all figures for the CVPR paper..."
python3 scripts/plot_figures.py

echo "======================================================================"
echo "Reproduction Complete!"
echo "Logs saved in: logs/"
echo "Figures saved in: figures/"
echo "======================================================================"
