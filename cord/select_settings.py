import argparse, collections, itertools, time
import numpy as np
from . import config as C
from .run import get_stream
from .metrics import evaluate
from .baselines_recent import ONNC, RioCPD
from common.utils import make_rng

SEED = 1000
ONNC_GRID = [(b, e, lr) for e in (1, 10) for b in (1, 10) for lr in (0.1, 0.01)]   # (batch, epochs, lr)
RIO_WINDOWS = (20, 50, 100)


def _evaluate(st, alarms):
    labels = np.cumsum(np.isin(np.arange(len(st['A'])), alarms))
    return evaluate(st, alarms, labels, C.TOL)


def onnc_best_f1(st, batch, epochs, lr, n_warm=300, lag=100):
    det = ONNC(st, make_rng(SEED, 'd', 'ONNC'), 1e9, lag=lag, batch=batch, epochs=epochs, lr=lr)
    scores, score_fn = [], det._score_and_train

    def recorded(ref, test):
        d = score_fn(ref, test); scores.append((det.t, d)); return d
    det._score_and_train = recorded
    for t in range(len(st['A'])):
        det.observe(st['S'][t], st['A'][t], st['R'][t], st['S2'][t])
    best = 0.0
    for h in C.GRIDS['ONNC']:                       # the alarm rule of ONNC.observe applied to the recorded scores
        sm = collections.deque(maxlen=lag + batch + 1); t0, alarms = 0, []
        for t, d in scores:
            sm.append(d)
            if t - t0 <= n_warm or len(sm) < sm.maxlen:
                continue
            if np.mean(sm) > h:
                sm.clear(); t0 = t; alarms.append(t)
        best = max(best, _evaluate(st, alarms)['f1'])
    return best


def rio_best_f1(st, w):
    best = 0.0
    for h in C.GRIDS['Rio-CPD']:
        det = RioCPD(st, make_rng(SEED, 'd', 'Rio'), h, w=w)
        alarms = [t for t in range(len(st['A'])) if det.observe(st['S'][t], st['A'][t], st['R'][t], st['S2'][t])]
        best = max(best, _evaluate(st, alarms)['f1'])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--detector', choices=['onnc', 'rio'], required=True)
    ap.add_argument('--envs', default=','.join(C.ENVS))
    args = ap.parse_args()
    envs = args.envs.split(',')
    settings = ONNC_GRID if args.detector == 'onnc' else RIO_WINDOWS
    rows = []
    for setting in settings:
        t0 = time.time(); f1 = []
        for env in envs:
            st = get_stream(env, SEED)
            f1.append(onnc_best_f1(st, *setting) if args.detector == 'onnc' else rio_best_f1(st, setting))
        rows.append((setting, f1))
        name = 'batch %d, epochs %d, lr %g' % setting if args.detector == 'onnc' else 'W = %d' % setting
        print(f"{name:28s} " + '  '.join(f'{e} {v:.2f}' for e, v in zip(envs, f1)) + f"  mean {np.mean(f1):.2f}   [{time.time() - t0:.0f} s]", flush=True)
    best = max(rows, key=lambda r: np.mean(r[1]))
    print('selected:', best[0])


if __name__ == '__main__':
    main()
