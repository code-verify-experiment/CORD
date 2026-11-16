"""Experimental configuration: benchmarks, stream design, model settings, detectors and threshold grids."""
from .baselines import MCUSUM, BOCPD, MBCD, LIFEWATCH, ScanB, ADWIN
from .cord import CORD
from .baselines_recent import WMNovelty, CUSUMRL, RioCPD, ONNC
from .baselines_simplified import RioCPDSimplified, ONNCSimplified

ENVS = ['cartpole', 'acrobot']
STREAM = dict(n_segments=8, seg_len=(1200, 2400), n_regimes=4, policy_switch_mean=1000, process_noise=1.0)
TOL = 300                       # detection tolerance (steps)


import numpy as _np
# Dynamics-model design (shared by CORD and MBCD so that only the test statistic differs).
# Selected on calibration seed 1000 by held-out prediction error.

MODEL = {
    'cartpole': dict(D=128, lengthscale=1.0),
    'acrobot': dict(D=512, lengthscale=1.5, ridge=0.01, target_dims=[4, 5],
                    in_scale=_np.array([1, 1, 1, 1, 2.0, 4.0])),
}

METHODS = {
    'CORD': (CORD, {}),
    'MCUSUM': (MCUSUM, {}),
    'BOCPD': (BOCPD, {}),
    'MBCD': (MBCD, {}),
    'LIFEWATCH': (LIFEWATCH, {}),
    'ScanB': (ScanB, {}),
    'ADWIN': (ADWIN, {}),
    # 2025-2026 detectors
    'WMNovelty': (WMNovelty, {}),
    'CUSUM-RL': (CUSUMRL, {}),
    'Rio-CPD': (RioCPD, {}),
    'ONNC': (ONNC, {}),
    'Rio-CPD-v1': (RioCPDSimplified, {}),     # simplified versions, only for Table S15
    'ONNC-v1': (ONNCSimplified, {}),
    # ablations of the proposed method
    'CORD-noGate': (CORD, dict(gate=False)),
    'CORD-noStd': (CORD, dict(standardize=False)),
    'CORD-noLib': (CORD, dict(library=False)),
    'CORD-warmQ': (CORD, dict(qbar_mode='warm_mean')),   # gate reference estimated during warm-up
}

# threshold grids searched on calibration seeds only (max mean F1)
GRIDS = {
    'CORD': [10, 20, 40, 80, 160, 320],
    'MCUSUM': [5, 10, 20, 40, 80, 160, 320],
    'BOCPD': [0.3, 0.5, 0.7, 0.9, 0.97, 0.995],
    'MBCD': [10, 20, 40, 80, 160, 320],
    'LIFEWATCH': [2, 3, 5, 8, 12, 20, 40],
    'ScanB': [2, 3, 5, 8, 12, 20, 40],
    'ADWIN': [1e-8, 1e-6, 1e-4, 0.002, 0.01, 0.05, 0.2],
}
BASELINES = ['MCUSUM', 'BOCPD', 'MBCD', 'LIFEWATCH', 'ScanB', 'ADWIN']
SOTA2025 = ['WMNovelty', 'CUSUM-RL', 'Rio-CPD', 'ONNC']
GRIDS.update({'WMNovelty': [0.1, 0.2, 0.3, 0.5, 0.7],
              'CUSUM-RL': [0.02, 0.05, 0.1, 0.2, 0.5],
              'Rio-CPD': [0.5, 1, 2, 3, 5, 8, 12, 20],
              'ONNC': [0.1, 0.25, 0.5, 1, 2, 4, 8, 16],
              'Rio-CPD-v1': [5, 10, 20, 40, 80],
              'ONNC-v1': [0.05, 0.1, 0.2, 0.3, 0.5]})
ABLATIONS = ['CORD-noGate', 'CORD-noStd', 'CORD-noLib', 'CORD-warmQ']
# every ablation is calibrated separately on the threshold grid of CORD
for _m in ABLATIONS:
    GRIDS[_m] = list(GRIDS['CORD'])


# Names shown in tables and figures (WM-Novelty and CUSUM-RL are adaptations, hence "-style").
DISPLAY = {'WMNovelty': 'WM-Novelty-style', 'CUSUM-RL': 'CUSUM-RL-style', 'ScanB': 'Scan-B',
           'Rio-CPD-v1': 'Rio-CPD (first version)', 'ONNC-v1': 'ONNC (first version)'}
