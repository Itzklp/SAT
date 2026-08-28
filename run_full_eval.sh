#!/bin/bash
set -e
echo "=== System A (LLM-only) ==="
python3 run_evaluation.py --system A --eval_file eval_test.json
echo "=== System B (vanilla RAG) ==="
python3 run_evaluation.py --system B --eval_file eval_test.json
echo "=== System E (full SAT) ==="
python3 run_evaluation.py --system E --eval_file eval_test.json
echo "=== ALL SYSTEMS COMPLETE ==="
