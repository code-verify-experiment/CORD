"""Robustness experiments.

  null       streams with NO environment change, only behaviour-policy switches:
             false-alarm rate of each detector and, for CORD variants, the empirical
             mean and variance of the in-control score u_t (Proposition 1).
  magnitude  smaller physical changes: every non-nominal regime perturbs one parameter
             by a relative amount eps in {0.1, 0.25, 0.5, 1.0}; no control reversals.
  noise      the original regimes with process-noise multipliers {0.5, 2, 4}
             (the main experiments use 1).
  learn_null    a learning agent (cord/agent.py) learns from scratch in one fixed regime: every
                alarm is false, and the behaviour changes only through learning.
  learn_change  the regime schedule of the main experiment, with the behaviour produced by the
                learning agent instead of the switching controllers.
  cord_sens     CORD with one setting halved or doubled at a time (allowance, clipping bound, gate
                temperature, warm-up, matching window, matching threshold, number of features,
                lengthscale, ridge penalty) on the main test streams.

All detectors use the thresholds calibrated in the main experiment (no re-tuning), and
the 15 test seeds 0-14. Usage:
  python -m cord.robustness --exp null,magnitude,noise,learn_null,learn_change,cord_sens --workers 4
  python -m cord.robustness --status
"""
import argparse, os, time
from multiprocessing import Pool
import numpy as np
from common import envs as E
from common.utils import make_rng, save_json, parse_seeds
from .streams import generate_stream, generate_learning_stream
from .metrics import evaluate
from .run import best_threshold
from . import config as C

import os as _os



OUT = ('results_gym' if _os.environ.get('CORD_BACKEND') == 'gymnasium' else 'results') + '/robust'
NULL_METHODS = ['CORD', 'CORD-noGate', 'CORD-noStd', 'MBCD', 'WMNovelty', 'ADWIN', 'MCUSUM', 'ScanB']
SENS_METHODS = ['CORD', 'WMNovelty', 'MBCD', 'ADWIN']
LEARN_METHODS = ['CORD', 'WMNovelty', 'MBCD', 'CUSUM-RL', 'ADWIN', 'MCUSUM', 'ScanB']
CORD_SENS = ['kappa0=0.25', 'kappa0=1', 'c=5', 'c=20', 'kappa=0.5', 'kappa=2', 'n_w=150', 'n_w=600',
             'n_m=75', 'n_m=300', 'm=0.5', 'm=2', 'D=half', 'D=double', 'ell=half', 'ell=double',
             'lambda=half', 'lambda=double']
EXPERIMENTS = ['null', 'magnitude', 'noise', 'learn_null', 'learn_change', 'cord_sens']
EPS = [0.1, 0.25, 0.5, 1.0]
NOISE = [0.5, 2.0, 4.0]


def scaled_regimes(env, eps):
    if env == 'cartpole':
        d = E.CARTPOLE_DEFAULT
        return [dict(), dict(length=d['length'] * (1 + eps)), dict(masspole=d['masspole'] * (1 + 2 * eps)),
                dict(force_mag=d['force_mag'] / (1 + eps)), dict(gravity=d['gravity'] * (1 + eps)),
                dict(masscart=d['masscart'] * (1 + 2 * eps))]
    d = E.ACROBOT_DEFAULT
    return [dict(), dict(m2=d['m2'] * (1 + eps)), dict(m1=d['m1'] * (1 + eps)), dict(l2=d['l2'] * (1 + eps / 2)),
            dict(g=d['g'] / (1 + eps)), dict(m2=d['m2'] / (1 + eps))]


def cord_variant(kw, setting):
    """Apply one setting of the sensitivity sweep to CORD's keyword arguments."""
    kw = dict(kw)
    name, value = setting.split('=')
    direct = {'kappa0': 'k', 'c': 'clip', 'kappa': 'kappa', 'n_w': 'n_warm', 'n_m': 'n_match', 'm': 'm_thr'}
    if name in direct:
        kw[direct[name]] = float(value) if name in ('kappa0', 'c', 'kappa', 'm') else int(value)
    else:
        key = {'D': 'D', 'ell': 'lengthscale', 'lambda': 'ridge'}[name]
        factor = 0.5 if value == 'half' else 2.0
        base = kw.get(key, {'D': 128, 'lengthscale': 1.0, 'ridge': 1.0}[key])
        kw[key] = int(base * factor) if key == 'D' else base * factor
    return kw


def _stream(exp, env, setting, seed):
    reg = E.CARTPOLE_REGIMES if env == 'cartpole' else E.ACROBOT_REGIMES
    saved = list(reg)
    try:
        kw = dict(C.STREAM)
        if exp in ('learn_null', 'learn_change'):
            kw.pop('policy_switch_mean', None)
            return generate_learning_stream(env, seed, no_change=(exp == 'learn_null'), **kw)
        if exp == 'null':
            kw['no_change'] = True
        elif exp == 'magnitude':
            reg[:] = scaled_regimes(env, setting)
        elif exp == 'noise':
            kw['process_noise'] = C.STREAM['process_noise'] * setting
        st = generate_stream(env, seed, **kw)
    finally:
        reg[:] = saved
    return st


def path(exp, env, setting, m, s):
    return f'{OUT}/{exp}/{env}/{setting}/{m}/seed{s}.json'


def job(args):
    exp, env, setting, m, s, h = args
    p = path(exp, env, setting, m, s)
    if os.path.exists(p):
        return
    st = _stream(exp, env, setting, s)
    cls, kw = C.METHODS[m]
    kw = dict(kw)
    if cls.__name__ in ('CORD', 'MBCD', 'WMNovelty', 'ADWIN'):
        kw.update(C.MODEL[env])
    if exp == 'cord_sens':
        kw = cord_variant(kw, setting)
    det = cls(st, make_rng(s, 'detector', m), h, **kw)
    dets, labels, us = [], np.empty(len(st['A']), int), []
    t0 = time.perf_counter()
    for t in range(len(st['A'])):
        x = (st['S'][t], st['A'][t], st['R'][t], st['S2'][t])
        if exp in ('null', 'learn_null') and hasattr(det, '_score') and det.collect is None and det.n_in >= det.n_warm:
            us.append(det._score(*x)[0])
        if det.observe(*x):
            dets.append(t)
        labels[t] = det.label
    r = evaluate(st, dets, labels, C.TOL)
    r.update(method=m, env=env, seed=s, exp=exp, setting=setting, h=h, T=len(st['A']),
             us_per_step=1e6 * (time.perf_counter() - t0) / len(st['A']))
    if 'episode_length' in st:
        L, E, T = st['episode_length'], st['episode_end'], len(st['A'])
        r.update(episode_len_first=float(L[E < 3000].mean()) if (E < 3000).any() else float('nan'),
                 episode_len_last=float(L[E >= T - 3000].mean()) if (E >= T - 3000).any() else float('nan'))
        if m == 'CORD':
            r.update(episode_end=[int(x) for x in E], episode_length=[int(x) for x in L])
    if us:
        u = np.asarray(us)
        r.update(u_mean=float(u.mean()), u_var=float(u.var()), u_tail5=float(np.mean(u > 5)))
    save_json(r, p)
    print(f'[{exp}] {env} {setting} {m} seed={s} alarms={r["n_alarms"]} F1={r["f1"]:.2f}', flush=True)


def jobs(exps, envs, seeds):
    out = []
    for exp in exps:
        settings = {'null': ['none'], 'magnitude': EPS, 'noise': NOISE, 'learn_null': ['none'],
                    'learn_change': ['agent'], 'cord_sens': CORD_SENS}[exp]
        methods = {'null': NULL_METHODS, 'magnitude': SENS_METHODS, 'noise': SENS_METHODS,
                   'learn_null': LEARN_METHODS, 'learn_change': LEARN_METHODS, 'cord_sens': ['CORD']}[exp]
        for env in envs:
            for m in methods:
                h = best_threshold(env, m)
                if h is None:
                    print(f'skip {exp}/{env}/{m}: threshold not calibrated yet (run python -m cord.run first)')
                    continue
                for setting in settings:
                    for s in seeds:
                        if not os.path.exists(path(exp, env, setting, m, s)):
                            out.append((exp, env, setting, m, s, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default=','.join(EXPERIMENTS))
    ap.add_argument('--envs', default=','.join(C.ENVS))
    ap.add_argument('--seeds', default='0-14')
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--status', action='store_true')
    a = ap.parse_args()
    J = jobs(a.exp.split(','), a.envs.split(','), parse_seeds(a.seeds))
    print(f'pending robustness jobs: {len(J)}')
    if a.status or not J:
        return
    if a.workers > 1:
        with Pool(a.workers) as p:
            p.map(job, J, chunksize=1)
    else:
        for j in J:
            job(j)


if __name__ == '__main__':
    main()
