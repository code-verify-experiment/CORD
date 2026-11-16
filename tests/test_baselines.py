#Unit tests of the Rio-CPD and ONNC implementations (python -m tests.test_baselines).
import numpy as np
from cord.baselines_recent import _Net, ONNC, RioCPD
from common.utils import make_rng


def test_gradient():
    rng = np.random.default_rng(0)
    net = _Net(7, (20, 20), rng, 0.01)
    X = rng.standard_normal((2, 7)); y = np.array([0.0, 1.0])
    def loss():
        f = net.predict(X)
        return -(np.log(1 - f[0])) - np.log(f[1])                 # paper Eq. (10) with n = 1
    gW, gb = net.grads(X, y, 1.0)
    worst, gmax = 0.0, 0.0
    for params, grads in ((net.W, gW), (net.b, gb)):
        for p, g in zip(params, grads):
            num = np.zeros_like(p)
            it = np.nditer(p, flags=['multi_index'])
            for _ in it:
                idx = it.multi_index; old = p[idx]
                p[idx] = old + 1e-6; lp = loss()
                p[idx] = old - 1e-6; lm = loss()
                p[idx] = old; num[idx] = (lp - lm) / 2e-6
            worst = max(worst, np.max(np.abs(num - g))); gmax = max(gmax, np.max(np.abs(g)))
    assert worst / gmax < 1e-6, worst / gmax
    print('network gradient vs finite differences: max abs error / largest gradient entry = %.2e  OK' % (worst / gmax))


def test_adam():
    rng = np.random.default_rng(1)
    net = _Net(3, (4,), rng, 0.1)
    p0 = [p.copy() for p in net.W + net.b]
    gW = [np.full_like(p, 0.3) for p in net.W]; gb = [np.full_like(p, 0.3) for p in net.b]
    net.adam_step(gW, gb)
    # first Adam step with constant gradient g: m_hat = g, v_hat = g^2, update = lr * g / (|g| + eps)
    for p, q in zip(p0, net.W + net.b):
        assert np.allclose(q, p - 0.1 * 0.3 / (0.3 + 1e-8), atol=1e-12)
    print('Adam first step equals lr * g / (|g| + eps)  OK')


def _spec():
    return dict(obs_dim=4, n_actions=2, obs_scale=np.array([2.4, 3.0, 0.21, 3.5]))


def _stream(kind, T=4000, tau=2000, seed=0):
    r = np.random.default_rng(seed); sc = _spec()['obs_scale']
    S = r.standard_normal((T, 4)) * sc * 0.4; A = r.integers(0, 2, T)
    inc = 0.04 * sc * r.standard_normal((T, 4)) + 0.02 * sc * (A[:, None] * 2 - 1) * np.array([1, 0, 0, 0])
    if kind == 'dynamics':
        inc[tau:, 0] += 0.2 * sc[0]
    if kind == 'correlation':                                      # correlation structure changes, means do not
        inc[tau:, 1] = inc[tau:, 0] * 3.0 + 0.2 * inc[tau:, 1]
    return S, A, S + inc


def _run(cls, kind, h, seed=0, **kw):
    S, A, S2 = _stream(kind, seed=seed)
    d = cls(_spec(), make_rng(seed, 'd'), h, **kw)
    return [t for t in range(len(A)) if d.observe(S[t], A[t], 0.0, S2[t])]


def test_behaviour():
    kws = {RioCPD: dict(w=100), ONNC: dict(batch=1, epochs=1, lr=0.01)}      # algorithmic check with the larger Rio-CPD window and gentle ONNC setting this test was written for (small windows give noisier correlation matrices)
    for cls, h, kinds in ((RioCPD, 2.0, ('correlation',)), (ONNC, 0.3, ('dynamics',))):
        for kind in kinds:
            al = _run(cls, kind, h, **kws[cls])
            pre = [a for a in al if a < 2000]; post = [a for a in al if a >= 2000]
            print(f'{cls.name}: change type "{kind}": {len(pre)} alarms before the change, first alarm after the change at '
                  f'+{(post[0] - 2000) if post else None} steps')
            assert post, f'{cls.name} did not detect the change'
        quiet = _run(cls, 'none', h, **kws[cls])
        print(f'{cls.name}: stationary stream: {len(quiet)} alarms')


if __name__ == '__main__':
    test_gradient(); test_adam(); test_behaviour()
