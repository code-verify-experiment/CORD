"""Monte Carlo check of Proposition 1 under Assumption 1.

Weights drawn from the prior N(0, sigma^2/lambda I) for every action and output coordinate, known
noise variance, exact Bayesian posterior, and actions chosen by policies that may use past errors.
The standardised errors should have mean 0, variance 1 and no lag-1 correlation for every policy.

    python verification/check_proposition1.py
"""
import numpy as np
rng=np.random.default_rng(0)
D,A,k,sig2,lam=24,2,2,0.05,1.0
W=rng.standard_normal((3,D)); b=rng.uniform(0,2*np.pi,D)
phi=lambda s: np.sqrt(2/D)*np.cos(s@W+b)
def run(policy,T=300,fixed_w=None):
    w=fixed_w if fixed_w is not None else rng.standard_normal((A,D,k))*np.sqrt(sig2/lam)
    P=np.stack([np.eye(D)/lam]*A); M=np.zeros((A,D,k)); s=rng.standard_normal(3); zs=[]; last=0.0
    for t in range(T):
        a=policy(s,last,t)
        f=phi(s); mu=f@M[a]; q=f@P[a]@f; v=sig2*(1+q)
        y=f@w[a]+np.sqrt(sig2)*rng.standard_normal(k)
        z=(y-mu)/np.sqrt(v); zs.append(z); last=float(np.abs(z).sum())
        Pf=P[a]@f; kk=Pf/(1+f@Pf); M[a]+=np.outer(kk,y-mu); P[a]-=np.outer(kk,Pf)
        s=np.tanh(s+0.3*np.r_[y,0.0]+0.1*rng.standard_normal(3))   # state depends on outcomes
    return np.array(zs)
policies={'random':lambda s,l,t:int(rng.integers(A)),
          'adaptive (acts on past error)':lambda s,l,t:int(l>1.0),
          'state-greedy (narrow coverage)':lambda s,l,t:int(s[0]>0.8)}
for name,pol in policies.items():
    Z=np.concatenate([run(pol) for _ in range(300)])
    u=((Z**2).sum(1)-k)/np.sqrt(2*k)
    zz=Z[:,0]; ac=np.corrcoef(zz[:-1],zz[1:])[0,1]
    print(f"{name:32s} zeta: mean {Z.mean():+.3f} var {Z.var():.3f} | u: mean {u.mean():+.3f} var {u.var():.3f} | lag-1 corr {ac:+.3f}")
# fixed environment (frequentist): variance < 1 early, -> 1 later
w0=rng.standard_normal((A,D,k))*np.sqrt(sig2/lam)
Zf=np.stack([run(policies['random'],fixed_w=w0) for _ in range(300)])
print("fixed beta*: var(zeta) over steps 0-20: %.3f, 20-100: %.3f, 200-300: %.3f; |mean| 0-20: %.3f"%(Zf[:,:20].var(),Zf[:,20:100].var(),Zf[:,200:].var(),abs(Zf[:,:20].mean())))
