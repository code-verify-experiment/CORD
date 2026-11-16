"""Classical online change-point detectors used as baselines.

All detectors share the interface observe(s, a, r, s2) -> bool (alarm) and the attribute `label`
(current segment label; detectors without a regime library issue a new label after every alarm).
All use the same warm-up protocol: after the start and after every alarm, `n_warm` transitions
re-estimate the in-control reference, and no alarm can be raised meanwhile.

  MCUSUM     two-sided CUSUM on every standardised coordinate of the transition vector (Page, 1954)
  BOCPD      Bayesian online change-point detection (Adams and MacKay, 2007)
  MBCD       CUSUM of log-likelihood ratios between dynamics models, with a model library
             (Alegre, Bazzan and da Silva, AAMAS 2021)
  LIFEWATCH  sliced Wasserstein window test with a memory of known distributions (Faber et al., 2022)
  ScanB      kernel MMD scan-B statistic (Li, Xie, Dai and Song, NeurIPS 2015)
  ADWIN      adaptive windowing (Bifet and Gavalda, SDM 2007) on the prediction error of the dynamics model"""


#########################################################################################################
import collections
import numpy as np
from scipy.special import gammaln, logsumexp
from common.dynamics import RFFDynamics


def transition_features(s, a, s2, scale, n_actions):
    oh = np.zeros(n_actions); oh[a] = 1.0
    return np.concatenate([np.asarray(s) / scale, oh, (np.asarray(s2) - np.asarray(s)) / scale * 10.0])


class _Base:
    has_library = False

    def __init__(self, spec, rng, h, n_warm=300):
        self.spec, self.rng, self.h, self.n_warm = spec, rng, h, n_warm
        self.t = 0
        self.label = 0
        self.next_label = 1
        self.warm = []

    def feat(self, s, a, s2):
        return transition_features(s, a, s2, self.spec['obs_scale'], self.spec['n_actions'])

    def _new_label(self):
        self.label = self.next_label; self.next_label += 1


class MCUSUM(_Base):
    name = 'MCUSUM'

    def __init__(self, spec, rng, h, k=0.5, n_warm=300):
        super().__init__(spec, rng, h, n_warm); self.k = k; self.mu = None

    def observe(self, s, a, r, s2):
        x = self.feat(s, a, s2); self.t += 1
        if self.mu is None:
            self.warm.append(x)
            if len(self.warm) >= self.n_warm:
                W = np.array(self.warm); self.mu, self.sd = W.mean(0), W.std(0) + 1e-3
                self.Sp = np.zeros_like(self.mu); self.Sm = np.zeros_like(self.mu); self.warm = []
            return False
        z = (x - self.mu) / self.sd
        self.Sp = np.maximum(0, self.Sp + z - self.k)
        self.Sm = np.maximum(0, self.Sm - z - self.k)
        if max(self.Sp.max(), self.Sm.max()) > self.h:
            self.mu = None; self._new_label(); return True
        return False


class BOCPD(_Base):
    name = 'BOCPD'

    def __init__(self, spec, rng, h, hazard=1 / 1000., r_max=300, r0=30, n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        self.H, self.r_max, self.r0 = hazard, r_max, r0
        self.mu0 = None

    def _reset_posterior(self):
        d = len(self.mu0)
        self.logR = np.array([0.0])
        self.m = np.zeros((1, d)); self.kap = np.ones((1, d)); self.al = np.ones((1, d)); self.be = np.ones((1, d))

    def observe(self, s, a, r, s2):
        x = self.feat(s, a, s2); self.t += 1
        if self.mu0 is None:
            self.warm.append(x)
            if len(self.warm) >= self.n_warm:
                W = np.array(self.warm); self.mu0, self.sd0 = W.mean(0), W.std(0) + 1e-3
                self.warm = []; self._reset_posterior(); self.since = 0
            return False
        z = (x - self.mu0) / self.sd0
        # Student-t predictive per run length (Normal-Gamma conjugate, independent dims)
        df = 2 * self.al
        sc2 = self.be * (self.kap + 1) / (self.al * self.kap)
        lp = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df * sc2)
              - (df + 1) / 2 * np.log1p((z - self.m) ** 2 / (df * sc2))).sum(1)
        grow = self.logR + lp + np.log(1 - self.H)
        cp = logsumexp(self.logR + lp + np.log(self.H))
        logR = np.concatenate([[cp], grow])
        logR -= logsumexp(logR)
        # posterior update
        m_new = (self.kap * self.m + z) / (self.kap + 1)
        kap_new = self.kap + 1
        al_new = self.al + 0.5
        be_new = self.be + self.kap * (z - self.m) ** 2 / (2 * (self.kap + 1))
        d = z.shape[0]
        self.m = np.vstack([np.zeros((1, d)), m_new]); self.kap = np.vstack([np.ones((1, d)), kap_new])
        self.al = np.vstack([np.ones((1, d)), al_new]); self.be = np.vstack([np.ones((1, d)), be_new])
        self.logR = logR
        if len(self.logR) > self.r_max:
            sl = slice(0, self.r_max)
            self.logR = self.logR[sl] - logsumexp(self.logR[sl])
            self.m, self.kap, self.al, self.be = self.m[sl], self.kap[sl], self.al[sl], self.be[sl]
        self.since += 1
        p_recent = float(np.exp(logsumexp(self.logR[:self.r0])))
        if self.since > 2 * self.r0 and p_recent > self.h:
            self.mu0 = None; self._new_label(); return True
        return False


class MBCD(_Base):
    """Model-based change detection (Alegre et al., 2021): CUSUM over the log-likelihood
    ratio between a model fitted on the recent window and the current model, with a
    library of models for recurring contexts. Uses the same RFF model class as CORD
    so that differences are attributable to the test statistic, not the model."""
    name = 'MBCD'
    has_library = True

    def __init__(self, spec, rng, h, window=200, refit_every=100, n_warm=300, D=128, lengthscale=1.0,
                 ridge=1.0, target_dims=None, in_scale=None):
        super().__init__(spec, rng, h, n_warm)
        self.window, self.refit_every, self.D = window, refit_every, D
        self.proto = RFFDynamics(spec['obs_dim'], spec['n_actions'], rng, n_features=D, obs_scale=spec['obs_scale'],
                                 lengthscale=lengthscale, ridge=ridge, target_dims=target_dims, in_scale=in_scale)
        self.cur = self.proto.copy(); self.cur.reset()
        self.cand = None
        self.buf = []
        self.lib = {}
        self.S = 0.0
        self.n_in = 0

    def observe(self, s, a, r, s2):
        self.t += 1
        tr = (s, a, r, s2)
        self.buf.append(tr)
        if len(self.buf) > self.window:
            self.buf.pop(0)
        if self.n_in < self.n_warm:
            self.cur.update(s, a, r, s2); self.n_in += 1
            return False
        if self.cand is None or self.t % self.refit_every == 0:
            self.cand = self.proto.copy(); self.cand.reset()
            self.cand.fit_batch(*map(np.array, zip(*self.buf[:-1])))
        _, q_c, nll_c, e, f = self.cur.score(s, a, r, s2)
        _, _, nll_n, _, _ = self.cand.score(s, a, r, s2)
        self.S = max(0.0, self.S + (nll_c - nll_n))
        if self.S > self.h:
            self.lib[self.label] = self.cur
            win = self.buf
            best, best_ll = None, -np.inf
            for lab, mdl in self.lib.items():
                ll = mdl.loglik(*map(np.array, zip(*win)))
                if ll > best_ll:
                    best, best_ll = lab, ll
            new = self.proto.copy(); new.reset()
            new.fit_batch(*map(np.array, zip(*win)))
            # MBCD switches to an existing model only if it explains the window at least as
            # well as a freshly fitted one; otherwise it instantiates a new model.
            if best is not None and best != self.label and best_ll >= new.loglik(*map(np.array, zip(*win))):
                self.label = best; self.cur = self.lib[best]
            else:
                self._new_label(); self.cur = new
            self.S = 0.0; self.n_in = 0
            return True
        self.cur.update(s, a, r, s2, f=f, e=e, q=q_c)
        return False


def sliced_w1(X, Y, P):
    """Sliced Wasserstein-1 distance with projection matrix P (d x n_proj)."""
    xp, yp = np.sort(X @ P, 0), np.sort(Y @ P, 0)
    q = np.linspace(0, 1, 50)
    return float(np.mean(np.abs(np.quantile(xp, q, axis=0) - np.quantile(yp, q, axis=0))))


class LIFEWATCH(_Base):
    name = 'LIFEWATCH'
    has_library = True

    def __init__(self, spec, rng, h, w=100, n_ref=300, every=10, n_proj=32, n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        self.w, self.n_ref, self.every = w, n_ref, every
        self.P = None; self.n_proj = n_proj
        self.ref = None; self.win = []; self.lib = {}

    def _calibrate(self, R):
        d = []
        for _ in range(20):
            idx = self.rng.permutation(len(R))
            d.append(sliced_w1(R[idx[:self.w]], R[idx[self.w:2 * self.w]], self.P))
        return np.mean(d), np.std(d) + 1e-6

    def observe(self, s, a, r, s2):
        x = self.feat(s, a, s2); self.t += 1
        if self.P is None:
            P = self.rng.standard_normal((len(x), self.n_proj)); self.P = P / np.linalg.norm(P, axis=0)
        self.win.append(x)
        if len(self.win) > self.w:
            self.win.pop(0)
        if self.ref is None:
            self.warm.append(x)
            if len(self.warm) >= max(self.n_warm, 2 * self.w):
                R = np.array(self.warm[-self.n_ref:]); self.ref = R
                self.mu0, self.sd0 = self._calibrate(R); self.warm = []
                # known-distribution check (recurrence) once a full new reference is available
                if self.lib.get('pending'):
                    self.lib.pop('pending')
                    best, bz = None, np.inf
                    for lab, (Rl, m0, s0) in self.lib.items():
                        z = (sliced_w1(R[-self.w:], Rl[:self.w], self.P) - m0) / s0
                        if z < bz:
                            best, bz = lab, z
                    if best is not None and bz < self.h / 2:
                        self.label = best
                self.lib[self.label] = (R, self.mu0, self.sd0)
            return False
        if self.t % self.every:
            return False
        z = (sliced_w1(self.ref, np.array(self.win), self.P) - self.mu0) / self.sd0
        if z > self.h:
            self.ref = None; self._new_label(); self.lib['pending'] = True; self.warm = []
            return True
        return False


class ScanB(_Base):
    name = 'ScanB'

    def __init__(self, spec, rng, h, w=60, B=3, every=10, n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        self.w, self.B, self.every = w, B, every
        self.pool = None; self.win = []

    def _mmd2(self, X, Y):
        def k(A, Bm):
            d = ((A[:, None, :] - Bm[None, :, :]) ** 2).sum(-1)
            return np.exp(-d / (2 * self.bw ** 2))
        n = len(X)
        Kxx, Kyy, Kxy = k(X, X), k(Y, Y), k(X, Y)
        np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
        return Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (n * (n - 1)) - 2 * Kxy.mean()

    def _stat(self, Y):
        idx = self.rng.choice(len(self.pool), size=(self.B, self.w), replace=False if len(self.pool) >= self.B * self.w else True)
        return np.mean([self._mmd2(self.pool[i], Y) for i in idx])

    def observe(self, s, a, r, s2):
        x = self.feat(s, a, s2); self.t += 1
        self.win.append(x)
        if len(self.win) > self.w:
            self.win.pop(0)
        if self.pool is None:
            self.warm.append(x)
            if len(self.warm) >= self.n_warm:
                P = np.array(self.warm); self.warm = []
                sub = P[self.rng.choice(len(P), min(200, len(P)), replace=False)]
                d = np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))
                self.bw = float(np.median(d[d > 0])) + 1e-6
                self.pool = P
                null = []
                for _ in range(20):
                    Y = P[self.rng.choice(len(P), self.w, replace=False)]
                    null.append(self._stat(Y))
                self.mu0, self.sd0 = np.mean(null), np.std(null) + 1e-9
            return False
        if self.t % self.every:
            return False
        z = (self._stat(np.array(self.win)) - self.mu0) / self.sd0
        if z > self.h:
            self.pool = None; self._new_label(); return True
        return False


class ADWIN(_Base):
    """ADWIN (Bifet and Gavalda, SDM 2007) applied to the prediction error of a dynamics model.

    As in error-based drift detection, the model (the random-feature model of CORD) is updated with
    every transition, and a drift detector watches its error. The monitored value is
    x_t = 1 - exp(-sum_d e_d^2 / (2 k sigma_d^2)) in [0, 1], where e is the one-step prediction error
    and sigma_d^2 the noise estimate of the model. ADWIN keeps a window of recent values; whenever the
    means of an older and a newer part of the window differ by more than

        eps_cut = sqrt(2 var / m * ln(2 / delta')) + 2 / (3 m) * ln(2 / delta'),
        1/m = 1/n0 + 1/n1,  delta' = delta / n,

    the older part is dropped. An alarm is raised only when the newer part has the larger mean
    (the error has increased); after an alarm the model and the window are reset. The window holds
    at most max_window values, cut points are checked every check_every steps at a spacing of
    check_every, and both parts must contain at least min_sub values. h is the confidence delta.
    """
    name = 'ADWIN'

    def __init__(self, spec, rng, h, D=128, lengthscale=1.0, ridge=1.0, target_dims=None, in_scale=None,
                 max_window=2000, check_every=10, min_sub=50, n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        self.proto = RFFDynamics(spec['obs_dim'], spec['n_actions'], rng, n_features=D, obs_scale=spec['obs_scale'],
                                 lengthscale=lengthscale, ridge=ridge, target_dims=target_dims, in_scale=in_scale)
        self.max_window, self.check_every, self.min_sub = max_window, check_every, min_sub
        self._reset()

    def _reset(self):
        self.model = self.proto.copy(); self.model.reset()
        self.win = collections.deque(maxlen=self.max_window)
        self.n_in = 0

    def _detect(self):
        while True:
            x = np.asarray(self.win); n = len(x)
            cuts = np.arange(self.min_sub, n - self.min_sub + 1, self.check_every)
            if len(cuts) == 0:
                return False
            cs = np.cumsum(x)
            mu_old = cs[cuts - 1] / cuts
            mu_new = (cs[-1] - cs[cuts - 1]) / (n - cuts)
            m = 1.0 / (1.0 / cuts + 1.0 / (n - cuts))
            ln = np.log(2.0 * n / self.h)
            eps = np.sqrt(2.0 * x.var() / m * ln) + 2.0 / (3.0 * m) * ln
            hit = np.flatnonzero(np.abs(mu_new - mu_old) > eps)
            if len(hit) == 0:
                return False
            j = hit[0]
            if mu_new[j] > mu_old[j]:
                return True
            for _ in range(int(cuts[j])):
                self.win.popleft()

    def observe(self, s, a, r, s2):
        self.t += 1
        _, q, _, e, f = self.model.score(s, a, r, s2)
        x = 1.0 - np.exp(-float(np.sum(e * e / self.model.sigma2)) / (2.0 * len(e)))
        self.model.update(s, a, r, s2, f=f, e=e, q=q)
        if self.n_in < self.n_warm:
            self.n_in += 1
            return False
        self.win.append(x)
        if self.t % self.check_every or len(self.win) < 2 * self.min_sub:
            return False
        if self._detect():
            self._reset(); self._new_label()
            return True
        return False

