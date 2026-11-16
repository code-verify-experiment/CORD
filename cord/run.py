"""Resumable runner for the calibration and test runs.

  python -m cord.run --budget 250        
  python -m cord.run --status

Pipeline: (1) per-seed calibration jobs for every (env, method, threshold) on the
calibration seeds; (2) once a method's calibration is complete its best
threshold (max mean F1) is frozen; (3) test jobs on the 15 test seeds.
Ablations are calibrated separately on the same grid. Every job writes one JSON file,
so the pipeline can be interrupted and resumed at any point.
"""
import argparse, os, time, glob
import numpy as np
from common.utils import make_rng, save_json, load_json, TEST_SEEDS, CALIB_SEEDS
from .streams import generate_stream
from .metrics import evaluate
from . import config as C

import os as _os
# separate result folder when the official Gymnasium environments are used (CORD_BACKEND=gymnasium)
OUT = 'results_gym' if _os.environ.get('CORD_BACKEND') == 'gymnasium' else 'results'
_STREAMS = {}


def get_stream(env, seed):
    if (env, seed) not in _STREAMS:
        _STREAMS[(env, seed)] = generate_stream(env, seed, **C.STREAM)
    return _STREAMS[(env, seed)]


def run_one(method, env, seed, h):
    st = get_stream(env, seed)
    cls, kw = C.METHODS[method]
    kw = dict(kw)
    if cls.__name__ in ('CORD', 'MBCD', 'WMNovelty', 'ADWIN'):
        kw.update(C.MODEL[env])
    det = cls(st, make_rng(seed, 'detector', method), h, **kw)
    dets, labels = [], np.empty(len(st['A']), int)
    t0 = time.perf_counter()
    for t in range(len(st['A'])):
        if det.observe(st['S'][t], st['A'][t], st['R'][t], st['S2'][t]):
            dets.append(t)
        labels[t] = det.label
    dt = time.perf_counter() - t0
    m = evaluate(st, dets, labels, C.TOL)
    m.update(us_per_step=1e6 * dt / len(st['A']), detections=dets, h=h, method=method, env=env, seed=seed,
             change_points=st['change_points'], policy_switches=st['policy_switches'], T=len(st['A']))
    return m


def calib_path(env, m, h, s): return f'{OUT}/calib/{env}/{m}/h{h}_seed{s}.json'
def test_path(env, m, s): return f'{OUT}/test/{env}/{m}/seed{s}.json'


def best_threshold(env, m):
    src = m
    f1 = {}
    for h in C.GRIDS[src]:
        ps = [calib_path(env, src, h, s) for s in CALIB_SEEDS]
        if not all(os.path.exists(p) for p in ps):
            return None
        f1[h] = np.mean([load_json(p)['f1'] for p in ps])
    return max(f1, key=f1.get)


def pending(envs, methods, seeds):
    jobs = []
    for env in envs:
        for m in methods:
            for h in C.GRIDS[m]:
                for s in CALIB_SEEDS:
                    if not os.path.exists(calib_path(env, m, h, s)):
                        jobs.append(('calib', env, m, h, s))
    for env in envs:
        for m in methods:
            h = best_threshold(env, m)
            if h is None:
                continue
            for s in seeds:
                tp = test_path(env, m, s)
                stale = (m in C.ABLATIONS and os.path.exists(tp) and load_json(tp).get('h') != h)
                if stale or not os.path.exists(tp):        # re-run results produced with an outdated threshold
                    jobs.append(('test', env, m, h, s))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', default=','.join(C.ENVS))
    ap.add_argument('--methods', default=','.join(['CORD'] + C.BASELINES + C.SOTA2025 + C.ABLATIONS))
    ap.add_argument('--seeds', default='0-14')
    ap.add_argument('--budget', type=float, default=1e9, help='wall-clock seconds')
    ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    from common.utils import parse_seeds
    envs, methods, seeds = a.envs.split(','), a.methods.split(','), parse_seeds(a.seeds)
    t0 = time.time()
    done = 0
    while True:
        jobs = pending(envs, methods, seeds)
        if a.status or not jobs:
            n_c = len([j for j in jobs if j[0] == 'calib']); n_t = len(jobs) - n_c
            print(f'pending: {n_c} calibration jobs, {n_t} test jobs (test jobs appear once calibration is complete)')
            return
        kind, env, m, h, s = jobs[0]
        r = run_one(m, env, s, h)
        save_json(r, calib_path(env, m, h, s) if kind == 'calib' else test_path(env, m, s))
        done += 1
        el = time.time() - t0
        print(f'[{kind}] {env} {m} h={h} seed={s} F1={r["f1"]:.2f} ({r["us_per_step"]:.0f}us/step) t={el:.0f}s', flush=True)
        if el > a.budget:
            print(f'budget reached after {done} jobs'); return


if __name__ == '__main__':
    main()
