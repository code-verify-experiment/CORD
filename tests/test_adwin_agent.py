#Unit tests of ADWIN and of the learning agent (python -m tests.test_adwin_agent).
import numpy as np
from cord.baselines import ADWIN
from cord.streams import generate_learning_stream
from cord import config as C

SPEC = dict(obs_dim=4, n_actions=2, obs_scale=np.ones(4))


def window_test(before, after, delta=0.002, seed=0):
    rng = np.random.default_rng(seed)
    det = ADWIN(SPEC, np.random.default_rng(1), delta)
    alarm_at = None
    for t, p in enumerate([before] * 1000 + [after] * 400):
        det.win.append(float(rng.random() < p))
        if t % det.check_every == 0 and len(det.win) >= 2 * det.min_sub and det._detect():
            alarm_at = t; break
    return alarm_at, len(det.win)


def test_adwin():
    t_up, _ = window_test(0.2, 0.5)
    assert t_up is not None and 1000 <= t_up < 1200, t_up
    print(f'ADWIN, error rises from 0.2 to 0.5 at step 1000: alarm at step {t_up}  OK')
    t_down, n_left = window_test(0.5, 0.2)
    assert t_down is None and n_left < 1400, (t_down, n_left)
    print(f'ADWIN, error falls from 0.5 to 0.2: no alarm, older part dropped (window {n_left} of 1400 values)  OK')
    t_flat, _ = window_test(0.3, 0.3)
    assert t_flat is None
    print('ADWIN, constant error: no alarm  OK')


def test_agent():
    kw = {k: v for k, v in C.STREAM.items() if k != 'policy_switch_mean'}
    for env, better in (('cartpole', np.greater), ('acrobot', np.less)):
        st = generate_learning_stream(env, 2000, no_change=True, **kw)
        L, E, T = st['episode_length'], st['episode_end'], len(st['A'])
        first, last = L[E < 3000].mean(), L[E >= T - 3000].mean()
        assert better(last, first), (env, first, last)
        print(f'agent learns on {env}: mean episode length {first:.0f} in the first and {last:.0f} in the last 3000 steps  OK')


if __name__ == '__main__':
    test_adwin()
    test_agent()
