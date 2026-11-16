# CORD: Covariate-gated Online Regime Detection

Code for the paper "Policy-Robust Online Detection of Environment Regime Changes in Reinforcement
Learning via Covariate-Gated Conditional Testing".

CORD detects changes of an environment's dynamics from the transitions of a reinforcement-learning
agent while remaining insensitive to changes of the agent's own behaviour. The package contains the
detector, nine baselines, the benchmark streams, every experiment of the paper, the analysis that
produces its tables and figures, and numerical checks of the theoretical results.

Pure Python (NumPy, SciPy, Matplotlib); no GPU and no simulator dependency. Tested with Python 3.9.6,
NumPy 2.0.2 and SciPy 1.13.1 on macOS 15.7.9 (Apple M4); supported on Python 3.9-3.12.


## Installation

``` bash
python3 -m venv .venv
source .venv/bin/activate          # for windows systems : .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick start

```bash
python -c "from cord.run import run_one; print(run_one('CORD', 'cartpole', 0, 20)['f1'])"
```

## Full reproduction

```bash
bash scripts/run_all.sh 4
```

All runners can be stopped and restarted; finished runs are skipped. Results are written to
`results/` (one JSON file per run).


For timing measurements, run with single-threaded numerical libraries:

```bash
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```
