"""Recent (2025-2026) detectors adapted to transition streams.

WMNovelty  WM-Novelty-style, after Zollicoffer et al. (ICML 2025). A transition is flagged when the
           likelihood of the observed increment under the random-feature dynamics model is below that
           under an uninformed Gaussian; an alarm is raised when the flagged fraction over 50 steps
           exceeds h. The original works in the latent space of a DreamerV2 world model; this is an
           adaptation of the idea.
CUSUMRL    CUSUM-RL-style, after Li, Shi, Wu and Fryzlewicz (Annals of Statistics 2025). CUSUM-type
           statistic over three split points of fitted Q-functions on a sliding window (linear
           fitted-Q iteration on random features). The original is an offline test for panel data with
           a sieve estimator and a multiplier bootstrap; here a calibrated threshold replaces the test.
RioCPD     Rio-CPD, Deng et al. (CIKM 2025), Section 3: correlation matrix B_t of the last W
           observations; Frechet mean of the previous matrices under the Log-Euclidean metric and the
           distance d_t of B_t to it; score D(t) = d_t - r_{t-1}, where r_{t-1} is the largest distance
           of a previous matrix to the mean; CUSUM y(t) = max(y(t-1) + D(t), 0); alarm when y(t) > rho,
           followed by a restart. W = 20 was chosen on calibration seed 1000 from {20, 50, 100}.
ONNC       Hushchyn, Arzymatov and Derkach (Machine Learning 115:56, 2026). A network with two hidden
           layers of 20 sigmoid units and a sigmoid output is trained online with Adam to separate the
           observation l = 100 steps ago (class 0) from the current one (class 1), with the
           cross-entropy loss of Eq. (10); the score is the symmetric Kullback-Leibler estimate of
           Eq. (11), computed from the predictions before the update, and is smoothed by a moving
           average over the lag. An alarm is raised when the smoothed score exceeds the threshold (the
           paper searches peaks over the whole series, which an online detector cannot do). Batch 10,
           one epoch and learning rate 0.1 were chosen on calibration seed 1000 from the grid of the
           paper (batch {1, 10}, epochs {1, 10}, learning rate {0.1, 0.01}).

All four use the transition vector of _Base (except the model-based WMNovelty) and the common
warm-up protocol.
"""
import collections
import numpy as np
from scipy.special import expit
from common.dynamics import RFFDynamics
from .baselines import _Base


class WMNovelty(_Base):
    name = 'WMNovelty'

    def __init__(self, spec, rng, h, win=50, n_warm=300, D=128, lengthscale=1.0, ridge=1.0, target_dims=None, in_scale=None):
        super().__init__(spec, rng, h, n_warm)
        self.proto = RFFDynamics(spec['obs_dim'], spec['n_actions'], rng, n_features=D, lengthscale=lengthscale,
                                 obs_scale=spec['obs_scale'], ridge=ridge, target_dims=target_dims, in_scale=in_scale)
        self.win = win; self._fresh()

    def _fresh(self):
        self.m = self.proto.copy(); self.m.reset(); self.n_in = 0; self.flags = []
        self.mu = None; self.buf = []

    def observe(self, s, a, r, s2):
        self.t += 1
        _, q, nll, e, f = self.m.score(s, a, r, s2)
        y = self.m.target(s, r, s2)
        if self.n_in < self.n_warm:
            self.m.update(s, a, r, s2, f=f, e=e, q=q); self.n_in += 1; self.buf.append(y)
            if self.n_in == self.n_warm:
                B = np.array(self.buf); self.mu, self.var = B.mean(0), B.var(0) + 1e-8
            return False
        nll0 = 0.5 * float(np.sum(np.log(2 * np.pi * self.var) + (y - self.mu) ** 2 / self.var))
        self.flags.append(float(nll > nll0))
        if len(self.flags) > self.win:
            self.flags.pop(0)
        if len(self.flags) == self.win and np.mean(self.flags) > self.h:
            self._fresh(); self._new_label(); return True
        if self.flags[-1] == 0.0:                     # learn only from non-novel transitions
            self.m.update(s, a, r, s2, f=f, e=e, q=q)
        return False


class CUSUMRL(_Base):
    name = 'CUSUM-RL'

    def __init__(self, spec, rng, h, window=1000, every=100, D=64, gamma=0.95, iters=5, n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        d = spec['obs_dim']
        self.W = rng.standard_normal((d, D)); self.b = rng.uniform(0, 2 * np.pi, D)
        self.D, self.window, self.every, self.gamma, self.iters = D, window, every, gamma, iters
        self.buf = []; self.since = 0

    def _phi(self, S):
        return np.sqrt(2.0 / self.D) * np.cos((np.asarray(S) / self.spec['obs_scale']) @ self.W + self.b)

    def _fqi(self, S, A, R, S2):
        nA = self.spec['n_actions']
        F, F2 = self._phi(S), self._phi(S2)
        w = np.zeros((nA, self.D))
        for _ in range(self.iters):
            y = R + self.gamma * (F2 @ w.T).max(1)
            for a in range(nA):
                idx = A == a
                if idx.sum() > 5:
                    Fa = F[idx]
                    w[a] = np.linalg.solve(Fa.T @ Fa + 1e-1 * np.eye(self.D), Fa.T @ y[idx])
        return w

    def observe(self, s, a, r, s2):
        self.t += 1; self.since += 1
        self.buf.append((s, a, r, s2))
        if len(self.buf) > self.window:
            self.buf.pop(0)
        if self.since < max(self.n_warm, self.window // 2) or self.t % self.every:
            return False
        S, A, R, S2 = map(np.array, zip(*self.buf))
        n = len(S); best = 0.0
        Fp = self._phi(S)
        for u in (n // 4, n // 2, 3 * n // 4):
            w1 = self._fqi(S[:u], A[:u], R[:u], S2[:u]); w2 = self._fqi(S[u:], A[u:], R[u:], S2[u:])
            diff = np.abs(Fp @ (w1 - w2).T).max(1).mean()
            best = max(best, np.sqrt(u * (n - u) / n) * diff / np.sqrt(n))
        if best > self.h:
            self.buf = []; self.since = 0; self._new_label(); return True
        return False


def _logm_spd(C):
    w, V = np.linalg.eigh(C)
    return (V * np.log(np.maximum(w, 1e-6))) @ V.T


class RioCPD(_Base):
    name = 'Rio-CPD'

    def __init__(self, spec, rng, h, w=20, lag=1, max_ref=500, eps=1e-3, n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        self.w, self.L, self.max_ref, self.eps = w, lag, max_ref, eps
        self.buf = None                  # ring buffer with the last w observations (the order does not matter for a correlation)
        self.cnt = 0
        self.logs = None                 # ring buffer of the vectorised log-correlation matrices
        self.n_ref = 0; self.pos = 0
        self.y = 0.0; self.t0 = 0        # CUSUM statistic and time of the last (re)start

    def _logcorr_vec(self, X):
        Xc = X - X.mean(0)
        sd = np.sqrt((Xc ** 2).mean(0))
        ok = sd > 1e-9
        Z = np.zeros_like(Xc); Z[:, ok] = Xc[:, ok] / sd[ok]
        C = Z.T @ Z / len(Z)             # Pearson correlation; constant coordinates become uncorrelated with variance 1
        C[~ok, ~ok] = 1.0
        L = _logm_spd(C + self.eps * np.eye(len(C)))
        return L[self.iu] * self.wt      # Euclidean norm of this vector = Frobenius norm of the matrix

    def _add(self, v):
        if self.logs is None:
            self.logs = np.zeros((self.max_ref, len(v)))
        self.logs[self.pos] = v
        self.pos = (self.pos + 1) % self.max_ref
        self.n_ref = min(self.n_ref + 1, self.max_ref)

    def observe(self, s, a, r, s2):
        x = self.feat(s, a, s2); self.t += 1
        if self.buf is None:
            m = len(x)
            self.buf = np.zeros((self.w, m)); self.iu = np.triu_indices(m)
            self.wt = np.where(self.iu[0] == self.iu[1], 1.0, np.sqrt(2.0))
        self.buf[self.cnt % self.w] = x; self.cnt += 1
        if self.cnt < self.w or self.t % self.L:
            return False
        v = self._logcorr_vec(self.buf)
        if self.n_ref == 0 or self.t - self.t0 <= self.n_warm:       # warm-up: build the reference set, no testing
            self._add(v)
            return False
        ref = self.logs[:self.n_ref]
        mean = ref.mean(0)                                           # Frechet mean under the Log-Euclidean metric
        radius = np.sqrt(((ref - mean) ** 2).sum(1)).max()           # r_{t-1}
        d = float(np.linalg.norm(v - mean))                          # d_t
        self.y = max(0.0, self.y + (d - radius))                     # D(t) = d_t - r_{t-1}
        self._add(v)                                                 # B_t joins the reference set (as in the paper)
        if self.y > self.h:
            self.y = 0.0; self.n_ref = 0; self.pos = 0; self.t0 = self.t
            self.cnt = 0; self._new_label()                          # restart: the window is refilled from scratch
            return True
        return False


class _InPlaceList(list):
    """list of arrays whose item assignment copies INTO the existing array (the arrays are views of one flat parameter vector)"""
    def __setitem__(self, i, v):
        list.__getitem__(self, i)[...] = v


class _Net:
    """Fully connected network with sigmoid activations (default 2 hidden layers of 20 units), Adam optimiser.
    Manual back-propagation; PyTorch's default initialisation U(-1/sqrt(fan_in), 1/sqrt(fan_in)) and Adam defaults.
    All parameters live in one flat vector (W and b are views), so one Adam step is a handful of vector operations."""

    def __init__(self, n_in, hidden, rng, lr, betas=(0.9, 0.999), eps=1e-8):
        sizes = [n_in] + list(hidden) + [1]
        Ws = [rng.uniform(-1, 1, (a, b)) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])]
        bs = [rng.uniform(-1, 1, b) / np.sqrt(a) for a, b in zip(sizes[:-1], sizes[1:])]
        arrs = Ws + bs
        self.theta = np.concatenate([p.ravel() for p in arrs])
        views, i = [], 0
        for p in arrs:
            views.append(self.theta[i:i + p.size].reshape(p.shape)); i += p.size
        self.W = _InPlaceList(views[:len(Ws)]); self.b = _InPlaceList(views[len(Ws):])
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = np.zeros_like(self.theta); self.v = np.zeros_like(self.theta)
        self.step_n = 0

    def forward(self, X):
        hs = [X]
        for W, b in zip(self.W, self.b):
            hs.append(expit(hs[-1] @ W + b))
        return hs

    def predict(self, X):
        return self.forward(X)[-1][:, 0]

    def grads(self, X, y, scale, hs=None):
        """gradient of the weighted cross-entropy; per-row weight `scale` (see ONNC); `hs` = activations of a forward pass"""
        hs = self.forward(X) if hs is None else hs
        f = hs[-1]
        g = (f - y[:, None]) * scale                      # d loss / d logit (sigmoid + cross-entropy)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for k in range(len(self.W) - 1, -1, -1):
            gW[k] = hs[k].T @ g
            gb[k] = g.sum(0)
            if k > 0:
                g = (g @ self.W[k].T) * hs[k] * (1.0 - hs[k])
        return gW, gb

    def adam_step(self, gW, gb):
        self.step_n += 1
        g = np.concatenate([p.ravel() for p in list(gW) + list(gb)])
        c1 = 1.0 - self.b1 ** self.step_n; c2 = 1.0 - self.b2 ** self.step_n
        self.m *= self.b1; self.m += (1 - self.b1) * g
        self.v *= self.b2; self.v += (1 - self.b2) * g * g
        self.theta -= self.lr * (self.m / c1) / (np.sqrt(self.v / c2) + self.eps)


class ONNC(_Base):
    name = 'ONNC'

    def __init__(self, spec, rng, h, lag=100, batch=10, epochs=1, lr=0.1, hidden=(20, 20), eps=1e-3,
                 smooth=None, scaler_n=None, loss='eq10', n_warm=300):
        super().__init__(spec, rng, h, n_warm)
        self.lag, self.n, self.epochs, self.eps = lag, batch, epochs, eps
        n_in = 2 * spec['obs_dim'] + spec['n_actions']
        self.net = _Net(n_in, hidden, rng, lr)
        self.loss_scale = 1.0 if loss == 'eq10' else 0.5          # eq10: sum of the two class means; 'bce_mean': mean over all rows
        self.hist = collections.deque(maxlen=lag + batch)          # observations needed to form (X(t-l), X(t))
        self.cache = collections.deque(maxlen=scaler_n or (lag + 2 * batch))   # rows of the running standardiser
        self.sm = collections.deque(maxlen=(smooth or (lag + batch)) + 1)      # causal moving average of the score
        self.t0 = 0

    def _score_and_train(self, ref, test):
        rows = np.vstack([ref, test])
        for r in rows:
            if len(self.cache) == self.cache.maxlen:
                self.cache.popleft()
            self.cache.append(r)
        C = np.asarray(self.cache); mu, sd = C.mean(0), C.std(0)
        Z = (rows - mu) / (sd + 1e-6)
        n = len(ref)
        hs = self.net.forward(Z)                                   # predictions BEFORE the update
        f = hs[-1][:, 0]
        fr, ft = f[:n], f[n:]
        e = self.eps
        d = np.mean(np.log(ft + e)) - np.mean(np.log(1 - ft + e)) + np.mean(np.log(1 - fr + e)) - np.mean(np.log(fr + e))
        y = np.concatenate([np.zeros(n), np.ones(n)])
        # gradient of  -(1/n) sum_ref log(1-f) - (1/n) sum_test log f   (paper Eq. 10)  ->  per-row weight 1/n
        for ep in range(self.epochs):
            gW, gb = self.net.grads(Z, y, self.loss_scale / n, hs if ep == 0 else None)
            self.net.adam_step(gW, gb)
        return float(d)

    def observe(self, s, a, r, s2):
        x = self.feat(s, a, s2); self.t += 1
        self.hist.append(x)
        if len(self.hist) < self.lag + self.n:
            return False
        ref = np.array([self.hist[i] for i in range(self.n)]); test = np.array([self.hist[-1 - i] for i in range(self.n - 1, -1, -1)])
        d = self._score_and_train(ref, test)
        self.sm.append(d)
        if self.t - self.t0 <= self.n_warm or len(self.sm) < self.sm.maxlen:
            return False
        if np.mean(self.sm) > self.h:
            self.sm.clear(); self.t0 = self.t; self._new_label()
            return True
        return False
