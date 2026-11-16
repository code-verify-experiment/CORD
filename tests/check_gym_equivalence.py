"""Checks that the native non-stationary CartPole/Acrobot and the official Gymnasium
environments (common.gym_envs) produce the same transitions in every regime.

  pip install gymnasium
  python -m tests.check_gym_equivalence

Both simulators are put in the same internal state, receive the same random actions (with
the regime's action map applied by each wrapper) and are compared after every step, both on
the internal 64-bit state and on the observation passed to the detectors; the process noise
is switched off. Differences of order 1e-12 or below are floating-point rounding.
"""
import numpy as np
from common.envs import make_env, CARTPOLE_REGIMES, ACROBOT_REGIMES
from common.gym_envs import GymNS


def check(name, regime, steps=3000, seed=0):
    rng = np.random.default_rng(seed)
    nat = make_env(name, np.random.default_rng(1), regime=regime, process_noise=0.0)
    gym = GymNS(name, np.random.default_rng(2), regime=regime, process_noise=0.0)
    nat.reset(); gym.reset()
    gym.env.state = np.array(nat.state, dtype=np.float64)
    worst, worst_state, n_term_mismatch = 0.0, 0.0, 0
    for _ in range(steps):
        a = int(rng.integers(nat.n_actions))
        o1, r1, t1, _ = nat.step(a)
        o2, r2, t2, _ = gym.step(a)
        worst = max(worst, float(np.max(np.abs(o1 - o2))), abs(r1 - r2))
        worst_state = max(worst_state, float(np.max(np.abs(np.asarray(nat.state, float) - np.asarray(gym.env.state, float)))))
        n_term_mismatch += int(t1 != t2)
        if t1 or t2 or nat.t >= 500:
            nat.reset(); gym.reset(); gym.t = 0
            gym.env.state = np.array(nat.state, dtype=np.float64)
    return worst, worst_state, n_term_mismatch


if __name__ == '__main__':
    import gymnasium
    print('gymnasium', gymnasium.__version__)
    ok = True
    for name, regs in [('cartpole', CARTPOLE_REGIMES), ('acrobot', ACROBOT_REGIMES)]:
        for k in range(len(regs)):
            w, ws, m = check(name, k)
            flag = 'OK' if (w < 1e-9 and ws < 1e-9 and m == 0) else 'DIFFERENT'
            ok &= flag == 'OK'
            print(f'{name:9s} regime R{k}: max |state diff| = {ws:.2e}, max |observation diff| = {w:.2e}, '
                  f'termination mismatches = {m}  {flag}')
    print('ALL IDENTICAL' if ok else 'SOME REGIMES DIFFER: report this output')
