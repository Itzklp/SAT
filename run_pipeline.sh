#!/bin/bash
set -e # Exit on error

echo ">>> 1. Installing Dependencies..."
pip install -r requirements.txt -q

echo ">>> 2. Generating Synthetic Data (Teacher Phase)..."
python generate_data.py

echo ">>> 3. Running Training Pipeline (Student Phase)..."
python train.py

echo ">>> SUCCESS! Your Amazon-Aligned Adapter is ready at: ./rufus_checkpoints/final_dpo_adapter"
