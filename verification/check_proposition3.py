"""Numerical checks for Proposition 3 (false-alarm bound with gating).

Computes theta* for k = 4 and k = 2, checks the convexity step Psi(theta* w) <= Psi(theta*)^w used in
the proof, and simulates the gated CUSUM with scores distributed as in Proposition 1 (no gate, a
random gate and an adversarial gate chosen from the past) to compare the empirical probability of a
false alarm within n steps with the bound n exp(-theta* h).

    python verification/check_propositions2_3.py
"""
import numpy as np
from scipy import stats, optimize, integrate
c,k0=10.0,0.5
def Psi(th,k):
    f=lambda x: np.exp(th*np.clip((x-k)/np.sqrt(2*k),-c,c))*stats.chi2.pdf(x,k)
    up=k+c*np.sqrt(2*k)
    return integrate.quad(f,0,up,limit=200)[0]+np.exp(th*c)*stats.chi2.sf(up,k)
for k in [4,2]:
    th=optimize.brentq(lambda t: np.log(Psi(t,k))-t*k0,1e-3,3)
    # convexity step used in the proof: Psi(th*w) <= Psi(th)^w
    ws=np.linspace(0,1,11); viol=max(Psi(th*w,k)-Psi(th,k)**w for w in ws)
    print(f"k={k}: theta*={th:.4f}; max violation of Psi(theta w)<=Psi(theta)^w: {viol:.2e}")
    rng=np.random.default_rng(1); R,n,h=20000,1000,12.0
    for name in ['no gate','random gate','adversarial gate']:
        S=np.zeros(R); hit=np.zeros(R,bool)
        for t in range(n):
            u=np.clip((rng.chisquare(k,R)-k)/np.sqrt(2*k),-c,c)
            if name=='no gate': w=np.ones(R)
            elif name=='random gate': w=rng.uniform(0,1,R)
            else: w=np.where(S>h/3,1.0,0.3)   # predictable: chosen from the past to push S up when already high
            S=np.maximum(0,S+w*u-k0); hit|=S>h
        print(f"   {name:17s} P(T<={n}) empirical {hit.mean():.4f}  bound n*exp(-theta h) = {n*np.exp(-th*h):.4f}")
