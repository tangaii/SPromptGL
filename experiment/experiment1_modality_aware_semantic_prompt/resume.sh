#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
trap 'python src/handoff.py' EXIT
python src/prepare.py
python -m unittest discover -s tests -v
python src/run_tasks.py --workers-per-device 6 --devices cuda:0,cuda:1
python src/aggregate.py
