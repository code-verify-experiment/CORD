"""Generation of piecewise-stationary transition streams with *distractor* policy shifts.

A stream is a sequence of transitions (s,a,r,s') produced by a behaviour policy acting in a CartPole-v1 / Acrobot-v1 environment whose physical regime switches at unknown times (the change points to detect). Independently, the behaviour policy itself switches at Poisson times: this changes the *marginal* distribution
of (s,a,s') but not the environment, so a correct detector must not fire.
"""
import numpy as np
from common.envs import make_env
from common.utils import make_rng


def _cartpole_policy(kind, obs, rng):
    gains = {0: (0.0, 0.0, 1.0, 0.5), 1: (0.5, 1.0, 10.0, 2.0), 3: (0.0, 0.0, 1.0, 0.5)}
    eps = {0: 0.05, 1: 0.10, 2: 1.0, 3: 0.40}[kind]
    if rng.random() < eps:
        return int(rng.integers(2))
    k = gains[kind]
    return int(np.dot(k, obs) > 0)


def _acrobot_policy(kind, obs, rng):
    eps = {0: 0.10, 1: 0.20, 2: 1.0, 3: 0.50}[kind]
    if rng.random() < eps:
        return int(rng.integers(3))
    v = obs[4] if kind in (0, 3) else obs[5]
    return 2 if v > 0 else 0


POLICIES = {'cartpole': _cartpole_policy, 'acrobot': _acrobot_policy}


def generate_stream(env_name, seed, n_segments=10, seg_len=(1500, 3000), n_regimes=4,
                    policy_switch_mean=1000, process_noise=1.0, n_policies=4, no_change=False,
                    backend=None):
    rng = make_rng(seed, 'stream', env_name)
    import os
    backend = backend or os.environ.get('CORD_BACKEND', 'native')
    if backend == 'gymnasium':
        from common.gym_envs import GymNS
        env = GymNS(env_name, make_rng(seed, 'env', env_name), regime=0, process_noise=process_noise)
    else:
        env = make_env(env_name, make_rng(seed, 'env', env_name), regime=0, process_noise=process_noise)
    pool = rng.choice(6, size=n_regimes, replace=False)
    regs = [int(rng.choice(pool))]
    for _ in range(n_segments - 1):
        regs.append(regs[-1] if no_change else int(rng.choice([r for r in pool if r != regs[-1]])))
    lens = rng.integers(seg_len[0], seg_len[1] + 1, size=n_segments)
    T = int(lens.sum())
    bounds = np.concatenate([[0], np.cumsum(lens)])
    regime_t = np.empty(T, int)
    for i in range(n_segments):
        regime_t[bounds[i]:bounds[i + 1]] = regs[i]
    change_points = [int(b) for b in bounds[1:-1] if not no_change]
    # behaviour-policy switches (distractors)
    pol_t = np.empty(T, int)
    t, cur = 0, int(rng.integers(n_policies))
    pol_switches = []
    while t < T:
        gap = int(rng.exponential(policy_switch_mean)) + 50
        pol_t[t:t + gap] = cur
        t += gap
        if t < T:
            cur = int(rng.choice([p for p in range(n_policies) if p != cur])); pol_switches.append(t)
    pfn = POLICIES[env_name]
    D = env.obs_dim
    S, S2 = np.empty((T, D)), np.empty((T, D))
    A, R, done = np.empty(T, int), np.empty(T), np.zeros(T, bool)
    env.set_regime(regime_t[0])
    obs = env.reset()
    for t in range(T):
        if env.regime != regime_t[t]:
            env.set_regime(regime_t[t])          # change happens mid-episode, unannounced
        a = pfn(pol_t[t], obs, rng)
        o2, r, term, trunc = env.step(a)
        S[t], A[t], R[t], S2[t] = obs, a, r, o2
        if term or trunc:
            done[t] = True
            obs = env.reset()
        else:
            obs = o2
    return dict(S=S, A=A, R=R, S2=S2, done=done, regime=regime_t, policy=pol_t,
                change_points=change_points, policy_switches=pol_switches,
                obs_scale=env.obs_scale, obs_dim=D, n_actions=env.n_actions, env=env_name)


def generate_learning_stream(env_name, seed, n_segments=10, seg_len=(1500, 3000), n_regimes=4,
                             process_noise=1.0, no_change=False, agent_kw=None):
    """Stream whose behaviour is produced by a learning agent (cord/agent.py).

    The regime schedule is drawn exactly as in generate_stream, so a seed gives the same regimes and
    change points in both kinds of stream. There are no policy switches: the behaviour changes
    through learning and decaying exploration, and in response to the regime changes themselves.
    The returned dictionary also contains the length of every completed episode and the step at
    which it ended.
    """
    from .agent import QLearningAgent
    rng = make_rng(seed, 'stream', env_name)
    env = make_env(env_name, make_rng(seed, 'env', env_name), regime=0, process_noise=process_noise)
    pool = rng.choice(6, size=n_regimes, replace=False)
    regs = [int(rng.choice(pool))]
    for _ in range(n_segments - 1):
        regs.append(regs[-1] if no_change else int(rng.choice([r for r in pool if r != regs[-1]])))
    lens = rng.integers(seg_len[0], seg_len[1] + 1, size=n_segments)
    T = int(lens.sum())
    bounds = np.concatenate([[0], np.cumsum(lens)])
    regime_t = np.empty(T, int)
    for i in range(n_segments):
        regime_t[bounds[i]:bounds[i + 1]] = regs[i]
    change_points = [int(b) for b in bounds[1:-1] if not no_change]
    agent = QLearningAgent(env.obs_scale, env.n_actions, make_rng(seed, 'agent', env_name), **(agent_kw or {}))
    D = env.obs_dim
    S, S2 = np.empty((T, D)), np.empty((T, D))
    A, R, done = np.empty(T, int), np.empty(T), np.zeros(T, bool)
    ep_len, ep_end, cur = [], [], 0
    env.set_regime(regime_t[0])
    obs = env.reset()
    for t in range(T):
        if env.regime != regime_t[t]:
            env.set_regime(regime_t[t])
        a = agent.act(obs)
        o2, r, term, trunc = env.step(a)
        agent.learn(obs, a, r, o2, term)
        S[t], A[t], R[t], S2[t] = obs, a, r, o2
        cur += 1
        if term or trunc:
            done[t] = True
            ep_len.append(cur); ep_end.append(t); cur = 0
            obs = env.reset()
        else:
            obs = o2
    return dict(S=S, A=A, R=R, S2=S2, done=done, regime=regime_t, policy=np.zeros(T, int),
                change_points=change_points, policy_switches=[], episode_length=np.array(ep_len),
                episode_end=np.array(ep_end), obs_scale=env.obs_scale, obs_dim=D, n_actions=env.n_actions,
                env=env_name)

