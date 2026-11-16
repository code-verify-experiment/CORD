import numpy as np


class QLearningAgent:
    def __init__(self, obs_scale, n_actions, rng, D=256, lengthscale=1.0, alpha=0.1, gamma=0.99,
                 eps_start=1.0, eps_end=0.05, eps_decay=5000):
        self.scale = np.asarray(obs_scale, float)
        self.n_actions, self.D = n_actions, D
        self.rng = rng
        self.W = rng.standard_normal((len(self.scale), D)) / lengthscale
        self.b = rng.uniform(0.0, 2.0 * np.pi, D)
        self.theta = np.zeros((n_actions, D))
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay
        self.t = 0

    def features(self, s):
        return np.sqrt(2.0 / self.D) * np.cos((np.asarray(s) / self.scale) @ self.W + self.b)

    def epsilon(self):
        frac = min(1.0, self.t / self.eps_decay)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def act(self, s):
        if self.rng.random() < self.epsilon():
            return int(self.rng.integers(self.n_actions))
        q = self.theta @ self.features(s)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def learn(self, s, a, r, s2, terminated):
        psi = self.features(s)
        target = r if terminated else r + self.gamma * float(np.max(self.theta @ self.features(s2)))
        self.theta[a] += self.alpha * (target - float(self.theta[a] @ psi)) * psi
        self.t += 1
