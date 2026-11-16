"""Choice of the predicted coordinates on Acrobot (Section 6.4 and Table S17).

Acrobot's observation is (cos t1, sin t1, cos t2, sin t2, w1, w2). CORD predicts only the increments of
the angular velocities w1 and w2. This script reproduces the evidence for that choice, using the
calibration seeds only:

  1. Calibration F1 of CORD on seeds 1000-1002 over the threshold grid, predicting all six increments
     or only the two angular-velocity increments.
  2. On seed 1000, per coordinate: held-out prediction error (1 - R^2) within each regime segment,
     the ratio of held-out to in-sample squared error (median and worst segment), and the ratio of the
     error on the next regime to the error on the same regime (how strongly the coordinate reacts to
     a change). The model is trained on the first 70% of each segment.

    python -m cord.select_model_outputs
"""
import numpy as np
from . import config as C
from .run import get_stream
from .cord import CORD
from .metrics import evaluate
from common.dynamics import RFFDynamics
from common.utils import make_rng, CALIB_SEEDS

NAMES = ['cos t1', 'sin t1', 'cos t2', 'sin t2', 'w1', 'w2']


def calibration_f1(target_dims):
    kw = {k: v for k, v in C.MODEL['acrobot'].items() if k != 'target_dims'}
    if target_dims is not None:
        kw['target_dims'] = target_dims
    rows = []
    for h in C.GRIDS['CORD']:
        f1, fa = [], []
        for s in CALIB_SEEDS:
            st = get_stream('acrobot', s)
            det = CORD(st, make_rng(s, 'detector', 'CORD'), h, **kw)
            alarms, labels = [], np.empty(len(st['A']), int)
            for t in range(len(st['A'])):
                if det.observe(st['S'][t], st['A'][t], st['R'][t], st['S2'][t]):
                    alarms.append(t)
                labels[t] = det.label
            m = evaluate(st, alarms, labels, C.TOL)
            f1.append(m['f1']); fa.append(m['fa_per_10k'])
        rows.append((h, float(np.mean(f1)), float(np.mean(fa))))
    return rows


def coordinate_errors(seed=1000, train_frac=0.7):
    st = get_stream('acrobot', seed)
    S, A, R, S2 = st['S'], st['A'], st['R'], st['S2']
    cps = [0] + list(st['change_points']) + [len(A)]
    mcfg = C.MODEL['acrobot']

    def model():
        return RFFDynamics(6, 3, make_rng(seed, 'selection'), n_features=mcfg['D'], lengthscale=mcfg['lengthscale'],
                           ridge=mcfg['ridge'], obs_scale=st['obs_scale'], in_scale=mcfg['in_scale'])

    def residuals(m, idx):
        Y = np.array([m.target(S[i], R[i], S2[i]) for i in idx])
        P = np.array([m.predict(S[i], A[i])[0] for i in idx])
        return Y, Y - P

    r2, ratio, react = [], [], []
    for j in range(len(cps) - 1):
        a, b = cps[j], cps[j + 1]
        cut = a + int(train_frac * (b - a))
        tr, te = np.arange(a, cut), np.arange(cut, b)
        m = model(); m.fit_batch(S[tr], A[tr], R[tr], S2[tr])
        _, res_in = residuals(m, tr)
        Y_te, res_te = residuals(m, te)
        r2.append((res_te ** 2).mean(0) / Y_te.var(0))
        ratio.append((res_te ** 2).mean(0) / (res_in ** 2).mean(0))
        if j + 2 < len(cps):
            c = cps[j + 2]
            nxt = np.arange(b + int(train_frac * (c - b)), c)
            Y_nx, res_nx = residuals(m, nxt)
            react.append(((res_nx ** 2).mean(0) / Y_nx.var(0)) / r2[-1])
    return np.array(r2), np.array(ratio), np.array(react)


def main():
    six, two = calibration_f1(None), calibration_f1([4, 5])
    print('Calibration F1 (seeds 1000-1002) and false alarms per 10^4 steps')
    print(f"{'h':>5s} {'six outputs':>18s} {'two velocity outputs':>22s}")
    for (h, f6, a6), (_, f2, a2) in zip(six, two):
        print(f"{h:5g} {f6:9.2f} ({a6:4.1f}) {f2:13.2f} ({a2:4.1f})")
    print(f"best mean F1: six outputs {max(r[1] for r in six):.2f}, two velocity outputs {max(r[1] for r in two):.2f}")
    r2, ratio, react = coordinate_errors()
    print('\nPer coordinate, seed 1000 (medians over regime segments)')
    print(f"{'coordinate':10s} {'1 - R^2':>8s} {'held-out / in-sample':>21s} {'worst segment':>14s} {'next / same regime':>19s}")
    for d, n in enumerate(NAMES):
        print(f"{n:10s} {np.median(r2[:, d]):8.3f} {np.median(ratio[:, d]):21.1f} {ratio[:, d].max():14.1f} {np.median(react[:, d]):19.2f}")


if __name__ == '__main__':
    main()
