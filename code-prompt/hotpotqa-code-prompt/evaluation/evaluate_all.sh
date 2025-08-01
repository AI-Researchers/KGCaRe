#!/bin/bash

# Usage: ./evaluate_all.sh path/to/prediction/folder path/to/gold_file.json
# Example: ./evaluate_all.sh ./predictions ./gold.json

PRED_DIR="$1"
GOLD_FILE="$2"
LOG_FILE="evaluation_log_500.txt"

if [ -z "$PRED_DIR" ] || [ -z "$GOLD_FILE" ]; then
  echo "Usage: $0 <prediction_folder> <gold_file>"
  exit 1
fi

echo "Evaluating predictions in folder: $PRED_DIR" > "$LOG_FILE"
echo "Using gold file: $GOLD_FILE" >> "$LOG_FILE"
echo "---------------------------------------------------" >> "$LOG_FILE"

for PRED_FILE in "$PRED_DIR"/*.json; do
  echo "Evaluating file: $PRED_FILE" | tee -a "$LOG_FILE"
  python3 eval_by_qtype.py "$PRED_FILE" "$GOLD_FILE" >> "$LOG_FILE" 2>&1
  echo "---------------------------------------------------" >> "$LOG_FILE"
done

echo "All evaluations completed. Results saved in $LOG_FILE."
