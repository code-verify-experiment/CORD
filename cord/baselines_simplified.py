"""First, simplified versions of Rio-CPD and ONNC (method keys Rio-CPD-v1 and ONNC-v1).

They are kept only for the comparison with the versions that follow the published algorithms
(cord/baselines_recent.py; Table S15 of the supplementary material).

RioCPDSimplified  CUSUM of the standardised Log-Euclidean distance between the correlation matrix of a
                  50-sample window and the Frechet mean of six reference windows.
ONNCSimplified    online logistic regression on quadratic features that separates samples from their
                  100-step-lagged counterparts; running divergence estimate as score.
"""
import numpy as np
from .baselines import _Base
from .baselines_recent import _logm_spd


class RioCPDSimplified(_Base):
    name = 'Rio-CPD-v1'

    def __init__(self, spec, rng, h, w=50, every=10, k=0.5, n_ref=6, n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        self.w, self.every, self.k, self.n_ref = w, every, k, n_ref
        self.win = []; self.refs = []; self.S = 0.0; self.calib = []; self.mu0 = None

    def _corr(self, X):
        C = np.corrcoef(np.asarray(X).T)
        C = np.nan_to_num(C, nan=0.0); np.fill_diagonal(C, 1.0)
        return _logm_spd(C + 1e-3 * np.eye(len(C)))

    def observe(self, s, a, r, s2):
        x = self.feat(s, a, s2); self.t += 1
        self.win.append(x)
        if len(self.win) > self.w:
            self.win.pop(0)
        if len(self.win) < self.w or self.t % self.every:
            return False
        L = self._corr(self.win)
        if len(self.refs) < self.n_ref:                          # reference windows (non-overlapping)
            if not self.refs or self.t - self.refs[-1][0] >= self.w:
                self.refs.append((self.t, L))
            return False
        mean = np.mean([R for _, R in self.refs], 0)              # Log-Euclidean Frechet mean
        d = float(np.linalg.norm(L - mean))
        if self.mu0 is None:
            self.calib.append(d)
            if len(self.calib) >= 10:
                self.mu0, self.sd0 = np.mean(self.calib), np.std(self.calib) + 1e-6
            return False
        self.S = max(0.0, self.S + (d - self.mu0) / self.sd0 - self.k)
        if self.S > self.h:
            self.refs, self.calib, self.mu0, self.S = [], [], None, 0.0
            self._new_label(); return True
        return False


class ONNCSimplified(_Base):
    name = 'ONNC-v1'

    def __init__(self, spec, rng, h, lag=100, avg=50, lr=0.05, n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        self.lag, self.avg, self.lr = lag, avg, lr
        self.hist = []; self.scores = []; self.wgt = None; self.n = 0

    def _f(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z @ self.wgt, -30, 30)))

    def _z(self, x):
        return np.concatenate([x, x * x, [1.0]])

    def observe(self, s, a, r, s2):
        x = self.feat(s, a, s2); self.t += 1; self.n += 1
        self.hist.append(x)
        if len(self.hist) > self.lag + 1:
            self.hist.pop(0)
        if len(self.hist) <= self.lag:
            return False
        z_new, z_old = self._z(x), self._z(self.hist[0])
        if self.wgt is None:
            self.wgt = np.zeros(len(z_new))
        p1, p0 = self._f(z_new), self._f(z_old)
        self.scores.append(np.log(max(p1, 1e-6)) + np.log(max(1 - p0, 1e-6)) + 2 * np.log(2))
        self.wgt += self.lr * ((1 - p1) * z_new - p0 * z_old)       # SGD on the logistic loss
        if len(self.scores) > self.avg:
            self.scores.pop(0)
        if self.n < self.n_warm + self.lag:
            return False
        if np.mean(self.scores) > self.h:
            self.hist, self.scores, self.wgt, self.n = [], [], None, 0
            self._new_label(); return True
        return False
