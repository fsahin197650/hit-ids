#!/usr/bin/env bash
# One command for everything. Watch it in the terminal:
#     bash scripts/run_all.sh
# If part 1 is already running in the background (started earlier with nohup),
# this just follows its log until it finishes, then runs part 2.
set -u
cd "$(dirname "$0")/.."
LOG1=results/logs/clean_pipeline.log
LOG2=results/logs/clean_pipeline_part2.log
if pgrep -f "scripts/run_clean_pipeline.sh" >/dev/null; then
  echo ">>> Part 1 is already running in the background — following $LOG1"
  PID=$(pgrep -of "scripts/run_clean_pipeline.sh")
  tail -n 30 -f --pid="$PID" "$LOG1"
else
  echo ">>> Starting part 1 (resumes automatically if partially done)"
  bash scripts/run_clean_pipeline.sh 2>&1 | tee -a "$LOG1"
fi
echo ">>> Starting part 2"
bash scripts/run_clean_pipeline_part2.sh 2>&1 | tee -a "$LOG2"
echo ">>> ALL DONE — results/paper_tables_clean.md is ready"
