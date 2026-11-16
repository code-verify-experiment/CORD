"""CORD: Covariate-gated Online Regime Detection.

Key idea. In RL the data distribution p(s,a,s') changes for two reasons: the
environment changes (what we must detect) or the agent's own behaviour changes
(what we must ignore). Only the *conditional* p(s'|s,a) identifies the environment.
CORD therefore tests the conditional, and protects the test against covariate
shift in (s,a) with three components:

 (C1) Heteroscedastic standardisation. A Bayesian RFF dynamics model provides a
      predictive variance v(s,a) = sigma^2 (1+q(s,a)). The standardised score
      u_t = (sum_d e_d^2/v_d - k)/sqrt(2k) has mean ~0 and variance ~1 under the
      null *wherever* (s,a) lies, so moving to new regions of the state space does
      not by itself bias the statistic.
 (C2) Epistemic gating. Where the model has little support (large q relative to the
      in-control average q_bar), residuals are uninformative about change and are
      down-weighted: w_t = exp(-kappa * max(0, q_t/q_bar - 1)).
 (C3) Robust CUSUM with change-time estimation. S_t = max(0, S_{t-1} + w_t*clip(u_t) - k).
      The last zero of S estimates the change time; held-out (suspicious) data is
      not used to update the model, avoiding absorption of the change.
 (C4) Regime library and recurrence identification. After an alarm, the post-change
      window is scored under every stored regime model; a model whose mean score
      is consistent with the null (u_bar < m_thr) is re-activated, otherwise a new
      regime model is created; the labels identify recurring regimes.
"""
import numpy as np
from common.dynamics import RFFDynamics


class CORD:
    name = 'CORD'
    has_library = True

    def __init__(self, spec, rng, h, k=0.5, kappa=1.0, clip=10.0, n_warm=300, n_match=150,
                 m_thr=1.0, D=128, lengthscale=1.0, gate=True, standardize=True, library=True,
                 include_reward=False, ridge=1.0, target_dims=None, in_scale=None, qbar_mode='ema'):
        self.spec, self.h, self.k, self.kappa, self.clip = spec, h, k, kappa, clip
        self.n_warm, self.n_match, self.m_thr = n_warm, n_match, m_thr
        self.gate, self.standardize, self.library = gate, standardize, library
        # qbar_mode: 'ema' (original) or 'warm_mean' (mean of q over the second half of each warm-up)
        self.qbar_mode = qbar_mode
        self._warm_q = []
        self.proto = RFFDynamics(spec['obs_dim'], spec['n_actions'], rng, n_features=D,
                                 lengthscale=lengthscale, obs_scale=spec['obs_scale'],
                                 include_reward=include_reward, ridge=ridge,
                                 target_dims=target_dims, in_scale=in_scale)
        self.model = self._fresh()
        self.lib = {}
        self.label, self.next_label = 0, 1
        self.t = 0
        self.S = 0.0
        self.qbar = None
        self.n_in = 0                # in-control samples seen by the current model
        self.suspect = []            # transitions since last zero of S
        self.collect = None          # post-alarm window
        self.last_change_est = 0
        self.alarm_times = []

    def _fresh(self):
        m = self.proto.copy(); m.reset(); return m

    # ------------------------------------------------------------------ #
    def _score(self, s, a, r, s2):
        u, q, nll, e, f = self.model.score(s, a, r, s2)
        if not self.standardize:      # ablation: homoscedastic residual (ignores epistemic inflation)
            v = self.model.sigma2
            z2 = e * e / v
            u = (z2.sum() - len(e)) / np.sqrt(2.0 * len(e))
        return u, q, e, f

    def _finish_collection(self):
        W = self.collect
        S_, A_, R_, S2_ = map(np.array, zip(*W))
        chosen = None
        if self.library and self.lib:
            scores = {lab: m.mean_score(S_, A_, R_, S2_) for lab, m in self.lib.items()}
            lab = min(scores, key=scores.get)
            if scores[lab] < self.m_thr:
                chosen = lab
        if chosen is not None:
            self.label = chosen
            self.model = self.lib[chosen].copy()
        else:
            self.label = self.next_label; self.next_label += 1
            self.model = self._fresh()
        for x in W:
            self.model.update(*x)
        self.n_in = len(W)
        self.collect = None
        self.S = 0.0
        self.suspect = []
        self._warm_q = []

    def observe(self, s, a, r, s2):
        self.t += 1
        tr = (s, a, r, s2)
        if self.collect is not None:                      # post-alarm: gather new-regime data
            self.collect.append(tr)
            if len(self.collect) >= self.n_match:
                self._finish_collection()
            return False
        if self.n_in < self.n_warm:                       # warm-up of a (new) regime model
            u, q, e, f = self._score(s, a, r, s2)
            self.model.update(s, a, r, s2, f=f, e=e, q=q)
            self.n_in += 1
            if self.qbar_mode == 'warm_mean':
                self._warm_q.append(q)
                if self.n_in >= self.n_warm:
                    self.qbar = float(np.mean(self._warm_q[len(self._warm_q) // 2:])); self._warm_q = []
                elif self.qbar is None:
                    self.qbar = q
            else:
                self.qbar = q if self.qbar is None else 0.99 * self.qbar + 0.01 * q
            return False
        u, q, e, f = self._score(s, a, r, s2)
        w = np.exp(-self.kappa * max(0.0, q / (self.qbar + 1e-12) - 1.0)) if self.gate else 1.0
        self.S = max(0.0, self.S + w * float(np.clip(u, -self.clip, self.clip)) - self.k)
        if self.S == 0.0:
            self.suspect = []
            self.last_change_est = self.t
        else:
            self.suspect.append(tr)
        if self.S > self.h:
            if self.library:
                self.lib[self.label] = self.model
            self.alarm_times.append(self.t)
            self.collect = list(self.suspect[-self.n_match:])
            if len(self.collect) >= self.n_match:
                self._finish_collection()
            return True
        if self.S < self.h / 4:                           # update only with in-control data
            self.model.update(s, a, r, s2, f=f, e=e, q=q)
            self.qbar = 0.995 * self.qbar + 0.005 * q
        return False
