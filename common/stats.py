
import numpy as np
from scipy import stats


def mean_ci(x, conf=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan
    m = x.mean()
    if n < 2:
        return m, 0.0
    h = stats.t.ppf((1 + conf) / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return m, h


def a12(x, y):
    """P(X > Y) + 0.5 P(X = Y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = (x[:, None] > y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return float(gt + 0.5 * eq)


def wilcoxon_paired(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    if np.allclose(d, 0):
        return 1.0
    nz = d[d != 0]
    exact = len(nz) == len(d) and len(np.unique(np.abs(nz))) == len(nz)
    try:
        return float(stats.wilcoxon(x, y, zero_method='wilcox', alternative='two-sided',
                                    method='exact' if exact else 'approx').pvalue)
    except ValueError:
        return 1.0


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def compare_to_proposed(results, proposed, higher_is_better=True):
    """results: {method: array over seeds}. Returns per-baseline test summary."""
    base = [m for m in results if m != proposed]
    raw = [wilcoxon_paired(results[proposed], results[m]) for m in base]
    adj = holm(raw) if base else []
    out = {}
    for m, p_raw, p_adj in zip(base, raw, adj):
        x, y = results[proposed], results[m]
        eff = a12(x, y) if higher_is_better else a12(-np.asarray(x), -np.asarray(y))
        out[m] = dict(p_raw=p_raw, p_holm=float(p_adj), a12=eff)
    return out
