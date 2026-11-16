import numpy as np

#CartPole Environment
CARTPOLE_DEFAULT = dict(gravity=9.8, masscart=1.0, masspole=0.1, length=0.5,
                        force_mag=10.0, action_map=(0, 1))

CARTPOLE_REGIMES = [
    dict(),                                                  # R0 nominal (Gymnasium)
    dict(length=1.0, masspole=0.3),                          # R1 long heavy pole
    dict(action_map=(1, 0)),                                 # R2 reversed controls
    dict(force_mag=5.0, length=0.3),                         # R3 weak actuator, short pole
    dict(gravity=4.9, masscart=2.0),                         # R4 low gravity, heavy cart
    dict(action_map=(1, 0), masspole=0.5, length=0.75),      # R5 reversed + heavy pole
]


class CartPoleNS:
    name = 'cartpole'
    obs_dim, n_actions, max_steps = 4, 2, 500
    obs_scale = np.array([2.4, 3.0, 0.21, 3.5])  # fixed scale used by learners

    def __init__(self, rng, regime=0, process_noise=0.0):
        self.rng = rng
        self.process_noise = process_noise
        self.tau = 0.02
        self.theta_thr = 12 * 2 * np.pi / 360
        self.x_thr = 2.4
        self.set_regime(regime)
        self.state = None
        self.t = 0

    def set_regime(self, regime):
        self.regime = regime
        p = dict(CARTPOLE_DEFAULT); p.update(CARTPOLE_REGIMES[regime])
        self.p = p

    def reset(self):
        self.state = self.rng.uniform(-0.05, 0.05, size=4)
        self.t = 0
        return self.state.copy()

    def step(self, action):
        p = self.p
        a = p['action_map'][int(action)]
        x, x_dot, th, th_dot = self.state
        total_mass = p['masspole'] + p['masscart']
        pml = p['masspole'] * p['length']
        force = p['force_mag'] if a == 1 else -p['force_mag']
        c, s = np.cos(th), np.sin(th)
        temp = (force + pml * th_dot ** 2 * s) / total_mass
        thacc = (p['gravity'] * s - c * temp) / (p['length'] * (4.0 / 3.0 - p['masspole'] * c ** 2 / total_mass))
        xacc = temp - pml * thacc * c / total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        th = th + self.tau * th_dot
        th_dot = th_dot + self.tau * thacc
        st = np.array([x, x_dot, th, th_dot])
        if self.process_noise > 0:
            st = st + self.process_noise * self.obs_scale * self.rng.standard_normal(4) * 0.01
        self.state = st
        self.t += 1
        terminated = bool(abs(st[0]) > self.x_thr or abs(st[2]) > self.theta_thr)
        truncated = self.t >= self.max_steps
        return st.copy(), 1.0, terminated, truncated



# Acrobot

ACROBOT_DEFAULT = dict(l1=1.0, l2=1.0, m1=1.0, m2=1.0, g=9.8, action_map=(0, 1, 2))

ACROBOT_REGIMES = [
    dict(),                                          # R0 nominal (Gymnasium)
    dict(m2=2.0),                                    # R1 heavy distal link
    dict(action_map=(2, 1, 0)),                      # R2 reversed torque
    dict(l1=0.75, l2=1.25),                          # R3 changed geometry
    dict(g=6.0, m1=1.5),                             # R4 low gravity, heavy proximal link
    dict(action_map=(2, 1, 0), m2=1.5),              # R5 reversed + heavy distal link
]


def _wrap(x, m, M):
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


class AcrobotNS:
    name = 'acrobot'
    obs_dim, n_actions, max_steps = 6, 3, 500
    obs_scale = np.array([1, 1, 1, 1, 4 * np.pi, 9 * np.pi])
    dt = 0.2
    MAX_VEL_1, MAX_VEL_2 = 4 * np.pi, 9 * np.pi
    TORQUES = (-1.0, 0.0, 1.0)

    def __init__(self, rng, regime=0, process_noise=0.0):
        self.rng = rng
        self.process_noise = process_noise
        self.set_regime(regime)
        self.state = None
        self.t = 0

    def set_regime(self, regime):
        self.regime = regime
        p = dict(ACROBOT_DEFAULT); p.update(ACROBOT_REGIMES[regime])
        p['lc1'], p['lc2'] = p['l1'] / 2.0, p['l2'] / 2.0   # Gymnasium: COM at half length
        self.p = p

    def _obs(self):
        s = self.state
        return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])

    def reset(self):
        self.state = self.rng.uniform(-0.1, 0.1, size=4)
        self.t = 0
        return self._obs()

    def _dsdt(self, s, a):
        p = self.p
        m1, m2, l1, lc1, lc2, g = p['m1'], p['m2'], p['l1'], p['lc1'], p['lc2'], p['g']
        I1 = I2 = 1.0
        th1, th2, dth1, dth2 = s
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(th2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(th2)) + I2
        phi2 = m2 * lc2 * g * np.cos(th1 + th2 - np.pi / 2.0)
        phi1 = (-m2 * l1 * lc2 * dth2 ** 2 * np.sin(th2) - 2 * m2 * l1 * lc2 * dth2 * dth1 * np.sin(th2)
                + (m1 * lc1 + m2 * l1) * g * np.cos(th1 - np.pi / 2) + phi2)
        ddth2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dth1 ** 2 * np.sin(th2) - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddth1 = -(d2 * ddth2 + phi1) / d1
        return np.array([dth1, dth2, ddth1, ddth2])

    def step(self, action):
        a = self.TORQUES[self.p['action_map'][int(action)]]
        s, dt = self.state, self.dt
        k1 = self._dsdt(s, a)
        k2 = self._dsdt(s + dt / 2 * k1, a)
        k3 = self._dsdt(s + dt / 2 * k2, a)
        k4 = self._dsdt(s + dt * k3, a)
        ns = s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if self.process_noise > 0:
            ns = ns + self.process_noise * 0.01 * np.array([1, 1, 4, 9]) * self.rng.standard_normal(4)
        ns[0] = _wrap(ns[0], -np.pi, np.pi)
        ns[1] = _wrap(ns[1], -np.pi, np.pi)
        ns[2] = np.clip(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = np.clip(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        self.t += 1
        terminated = bool(-np.cos(ns[0]) - np.cos(ns[1] + ns[0]) > 1.0)
        reward = 0.0 if terminated else -1.0
        truncated = self.t >= self.max_steps
        return self._obs(), reward, terminated, truncated


ENVS = {'cartpole': CartPoleNS, 'acrobot': AcrobotNS}


def make_env(name, rng, regime=0, process_noise=0.0):
    return ENVS[name](rng, regime=regime, process_noise=process_noise)
