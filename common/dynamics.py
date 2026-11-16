import numpy as np


class RFFDynamics:
    def __init__(self, obs_dim, n_actions, rng, n_features=128, lengthscale=1.0,
                 ridge=1.0, obs_scale=None, include_reward=False, noise_floor=1e-6, target_dims=None, in_scale=None):
        self.obs_dim, self.n_actions, self.D = obs_dim, n_actions, n_features
        # target_dims: state dimensions whose increments are modelled (default: all).
        self.tdims = np.arange(obs_dim) if target_dims is None else np.asarray(target_dims)
        self.out_dim = len(self.tdims) + (1 if include_reward else 0)
        self.include_reward = include_reward

        ######################################################################
        self.scale = np.ones(obs_dim) if obs_scale is None else np.asarray(obs_scale, float)
        self.in_scale = self.scale if in_scale is None else np.asarray(in_scale, float)
        self.W = rng.standard_normal((obs_dim, n_features)) / lengthscale
        self.b = rng.uniform(0, 2 * np.pi, n_features)
        self.ridge = ridge
        self.noise_floor = noise_floor
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self):
        D, A, O = self.D, self.n_actions, self.out_dim
        self.P = np.stack([np.eye(D) / self.ridge for _ in range(A)])
        self.M = np.zeros((A, D, O))
        self.sq = np.full(O, 1e-2)       # running mean of standardised squared residuals * sigma^2
        self.n_res = 0
        self.n = np.zeros(A, int)

    def copy(self):
        c = object.__new__(RFFDynamics)
        c.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return c

    # ------------------------------------------------------------------ #
    def phi(self, s):
        x = np.asarray(s) / self.in_scale
        return np.sqrt(2.0 / self.D) * np.cos(x @ self.W + self.b)

    def target(self, s, r, s2):
        d = ((np.asarray(s2) - np.asarray(s)) / self.scale)[self.tdims]
        return np.append(d, r) if self.include_reward else d

    @property
    def sigma2(self):
        return np.maximum(self.sq, self.noise_floor)

    def predict(self, s, a):
        f = self.phi(s)
        mu = f @ self.M[a]
        q = float(f @ self.P[a] @ f)
        return mu, q, f

    def predict_mean_batch(self, S, A):
        F = self.phi(np.asarray(S))
        out = np.zeros((len(F), self.out_dim))
        for a in np.unique(A):
            idx = A == a
            out[idx] = F[idx] @ self.M[a]
        return out

    def score(self, s, a, r, s2):
        """Returns (standardised chi-square score u, epistemic q, Gaussian NLL)."""
        mu, q, f = self.predict(s, a)
        e = self.target(s, r, s2) - mu
        v = self.sigma2 * (1.0 + q)
        z2 = e * e / v
        k = len(e)
        u = (z2.sum() - k) / np.sqrt(2.0 * k)
        nll = 0.5 * float(np.sum(np.log(2 * np.pi * v) + z2))
        return float(u), q, nll, e, f

    def update(self, s, a, r, s2, f=None, e=None, q=None):
        if f is None:
            mu, q, f = self.predict(s, a)
            e = self.target(s, r, s2) - mu
        P = self.P[a]
        Pf = P @ f
        denom = 1.0 + float(f @ Pf)
        k = Pf / denom
        self.M[a] += np.outer(k, e)
        P -= np.outer(k, Pf)
        # aleatoric noise estimate from one-step-ahead residuals (unbiased under Gaussian BLR)
        self.n_res += 1
        lr = max(1.0 / self.n_res, 2e-3)
        self.sq = (1 - lr) * self.sq + lr * (e * e) / (1.0 + q)
        self.n[a] += 1

    def fit_batch(self, S, A, R, S2):
        """Closed-form posterior from a batch (equivalent to sequential updates from the prior)."""
        S, A, R, S2 = map(np.asarray, (S, A, R, S2))
        F = self.phi(S)                                   # (n, D)
        Y = np.stack([self.target(S[i], R[i], S2[i]) for i in range(len(S))])
        rss, dof, n_tot = np.zeros(self.out_dim), 0.0, 0
        for a in range(self.n_actions):
            idx = A == a
            if not idx.any():
                continue
            Fa, Ya = F[idx], Y[idx]
            G = Fa.T @ Fa + self.ridge * np.eye(self.D)
            P = np.linalg.inv(G)
            self.P[a] = P
            self.M[a] = P @ (Fa.T @ Ya)
            E = Ya - Fa @ self.M[a]
            rss += (E * E).sum(0)
            dof += float(np.trace(Fa @ P @ Fa.T))
            n_tot += int(idx.sum()); self.n[a] += int(idx.sum())
        if n_tot > dof + 1:
            self.sq = rss / (n_tot - dof)
            self.n_res = n_tot

    def mean_score(self, S, A, R, S2):
        """Average standardised score over a window (used for regime matching)."""
        if len(S) == 0:
            return np.inf
        return float(np.mean([self.score(S[i], A[i], R[i], S2[i])[0] for i in range(len(S))]))

    def loglik(self, S, A, R, S2):
        return -float(np.sum([self.score(S[i], A[i], R[i], S2[i])[2] for i in range(len(S))]))
