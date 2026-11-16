#Evaluation metrics for online change-point detection.
import numpy as np


def match_detections(cps, dets, tol):
    """Each true change point cp is detected by the first alarm in [cp, cp+tol].
    Unmatched alarms are false alarms. Returns (tp, fp, fn, delays)."""
    dets = sorted(dets)
    used = set()
    delays = []
    for cp in cps:
        hit = [d for d in dets if cp <= d <= cp + tol and d not in used]
        if hit:
            used.add(hit[0]); delays.append(hit[0] - cp)
    tp = len(delays)
    fp = len([d for d in dets if d not in used])
    fn = len(cps) - tp
    return tp, fp, fn, delays


def adjusted_rand_index(a, b):
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    C = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(C, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2.0
    sc = comb(C).sum()
    sa, sb = comb(C.sum(1)).sum(), comb(C.sum(0)).sum()
    n = comb(len(a))
    exp = sa * sb / n
    mx = 0.5 * (sa + sb)
    return float((sc - exp) / (mx - exp)) if mx != exp else 0.0


def evaluate(stream, dets, labels, tol, false_alarm_window=None):
    cps = stream['change_points']
    T = len(stream['A'])
    tp, fp, fn, delays = match_detections(cps, dets, tol)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # false alarms attributable to behaviour-policy switches (alarm within tol after a policy switch
    # and not within tol after an environment change)
    ps = stream['policy_switches']
    pol_fa = 0
    for d in dets:
        near_cp = any(cp <= d <= cp + tol for cp in cps)
        near_ps = any(p <= d <= p + tol for p in ps)
        pol_fa += int(near_ps and not near_cp)
    ari = adjusted_rand_index(stream['regime'], labels) if labels is not None else np.nan
    return dict(f1=f1, precision=prec, recall=rec,
                delay=float(np.mean(delays)) if delays else float('nan'),
                fa_per_10k=1e4 * fp / T, policy_fa_per_10k=1e4 * pol_fa / T,
                ari=ari, n_alarms=len(dets))
