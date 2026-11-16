#!/bin/bash

#   bash scripts/run_all.sh [WORKERS]      (WORKERS = parallel processes for the robustness runs, default 4)
set -e
cd "$(dirname "$0")/.."
W=${1:-4}
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m tests.test_baselines
python -m tests.test_adwin_agent
python -m cord.run
python -m cord.run --methods Rio-CPD-v1,ONNC-v1
python -m cord.robustness --workers "$W"
python -m cord.select_settings --detector rio | tee select_settings_rio.txt
python -m cord.select_settings --detector onnc | tee select_settings_onnc.txt
python -m cord.select_model_outputs | tee select_model_outputs.txt
python verification/check_proposition1.py | tee check_proposition1.txt
python verification/check_proposition3.py | tee check_proposition3.txt
python -m cord.timing_check | tee timing_check.txt

