
import numpy as np
from .envs import (CARTPOLE_DEFAULT, CARTPOLE_REGIMES, ACROBOT_DEFAULT, ACROBOT_REGIMES,
                   CartPoleNS, AcrobotNS, _wrap)


def _acrobot_class():
    from gymnasium.envs.classic_control.acrobot import AcrobotEnv

    class AcrobotG(AcrobotEnv):
        gravity = 9.8

        def _dsdt(self, s_augmented):
            m1, m2 = self.LINK_MASS_1, self.LINK_MASS_2
            l1 = self.LINK_LENGTH_1
            lc1, lc2 = self.LINK_COM_POS_1, self.LINK_COM_POS_2
            I1 = I2 = self.LINK_MOI
            g = self.gravity
            a = s_augmented[-1]
            s = s_augmented[:-1]
            theta1, theta2, dtheta1, dtheta2 = s
            d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
            d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
            phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.0)
            phi1 = (-m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2)
                    - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
                    + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2)
            if self.book_or_nips == "nips":
                ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
            else:
                ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(theta2) - phi2) \
                    / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
            ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
            return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    return AcrobotG


class GymNS:
    """Non-stationary wrapper around the official Gymnasium environment."""

    def __init__(self, name, rng, regime=0, process_noise=0.0):
        import gymnasium as gym
        self.name, self.rng, self.process_noise = name, rng, process_noise
        if name == 'cartpole':
            self.env = gym.make('CartPole-v1').unwrapped
            self.obs_dim, self.n_actions, self.obs_scale = 4, 2, CartPoleNS.obs_scale
        elif name == 'acrobot':
            self.env = _acrobot_class()()
            self.obs_dim, self.n_actions, self.obs_scale = 6, 3, AcrobotNS.obs_scale
        else:
            raise ValueError(name)
        self.max_steps = 500
        self.env.reset(seed=int(rng.integers(2 ** 31 - 1)))
        self.set_regime(regime)
        self.t = 0

    def set_regime(self, regime):
        self.regime = regime
        e = self.env
        if self.name == 'cartpole':
            p = dict(CARTPOLE_DEFAULT); p.update(CARTPOLE_REGIMES[regime])
            e.gravity, e.masscart, e.masspole = p['gravity'], p['masscart'], p['masspole']
            e.length, e.force_mag = p['length'], p['force_mag']
            e.total_mass = e.masspole + e.masscart              # cached by Gymnasium: recompute
            e.polemass_length = e.masspole * e.length            # cached by Gymnasium: recompute
        else:
            p = dict(ACROBOT_DEFAULT); p.update(ACROBOT_REGIMES[regime])
            e.LINK_LENGTH_1, e.LINK_LENGTH_2 = p['l1'], p['l2']
            e.LINK_MASS_1, e.LINK_MASS_2 = p['m1'], p['m2']
            e.LINK_COM_POS_1, e.LINK_COM_POS_2 = p['l1'] / 2.0, p['l2'] / 2.0
            e.gravity = p['g']
        self.action_map = tuple(p['action_map'])

    def _obs(self):
        s = np.array(self.env.state, dtype=np.float64)
        if self.name == 'cartpole':
            return s
        # Gymnasium's Acrobot._get_ob() returns float32; the same formula is evaluated here in
        # float64 from Gymnasium's internal (float64) state, so no precision is lost.
        return np.array([np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]])

    def reset(self):
        self.env.reset()
        self.t = 0
        return self._obs()

    def step(self, action):
        e = self.env
        _, r, terminated, _, _ = e.step(self.action_map[int(action)])
        if self.process_noise > 0:
            if self.name == 'cartpole':
                st = np.array(e.state, dtype=np.float64)
                st = st + self.process_noise * self.obs_scale * self.rng.standard_normal(4) * 0.01
                e.state = st
                terminated = bool(abs(st[0]) > e.x_threshold or abs(st[2]) > e.theta_threshold_radians)
                r = 1.0
            else:
                ns = np.array(e.state, dtype=np.float64)
                ns = ns + self.process_noise * 0.01 * np.array([1, 1, 4, 9]) * self.rng.standard_normal(4)
                ns[0] = _wrap(ns[0], -np.pi, np.pi); ns[1] = _wrap(ns[1], -np.pi, np.pi)
                ns[2] = np.clip(ns[2], -e.MAX_VEL_1, e.MAX_VEL_1); ns[3] = np.clip(ns[3], -e.MAX_VEL_2, e.MAX_VEL_2)
                e.state = ns
                terminated = bool(e._terminal())
                r = 0.0 if terminated else -1.0
        self.t += 1
        return self._obs(), float(r), bool(terminated), self.t >= self.max_steps
