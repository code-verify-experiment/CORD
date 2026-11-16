import json, os, time, hashlib
import numpy as np

TEST_SEEDS = list(range(15))          # test seeds
CALIB_SEEDS = [1000, 1001, 1002]      # calibration seeds (threshold selection only)


def parse_seeds(text):
    #'0-14' -> [0..14]; '0,3,5' -> [0,3,5]
    out = []
    for part in str(text).split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def make_rng(seed, *streams):
    ss = np.random.SeedSequence([int(seed)] + [int(hashlib.md5(str(s).encode()).hexdigest()[:8], 16) for s in streams])
    return np.random.default_rng(ss)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.bool_,)): return bool(o)
        raise TypeError(type(o))
    with open(path, 'w') as f:
        json.dump(obj, f, default=conv, indent=1)


def load_json(path):
    with open(path) as f:
        return json.load(f)
