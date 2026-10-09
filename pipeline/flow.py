"""Cloud motion between two 512x256 cloud maps (numpy port of the app's FlowEstimator).
Block matching (17x17 templates on a 64x32 grid, +-7 px search), sub-pixel refinement, outlier median,
gap fill, smoothing. Output: 64x32 RG8, degrees/hour (east, north), 128 = still, FLOW_MAX = 2.5 deg/h.
"""
import numpy as np
from scipy import ndimage

W, H, GW, GH, HALF, SEARCH, FLOW_MAX = 512, 256, 64, 32, 8, 7, 2.5

def estimate(a, b, hours):
    """a, b: float32 (256, 512) in 0..1, older then newer."""
    if not (0.15 <= hours <= 9):
        return still()
    cell = W // GW
    cy = np.arange(GH) * cell + cell // 2
    cx = np.arange(GW) * cell + cell // 2
    S = 2 * SEARCH + 1
    cost = np.zeros((S, S, GH, GW), np.float32)
    k = 2 * HALF + 1
    for sy in range(-SEARCH, SEARCH + 1):
        bs = np.roll(b, -sy, axis=0)
        valid = np.ones(H, bool)
        if sy > 0: valid[H - sy:] = False
        if sy < 0: valid[:-sy] = False
        for sx in range(-SEARCH, SEARCH + 1):
            d = np.abs(a - np.roll(bs, -sx, axis=1))
            d[~valid] = 0.25 / k                                    # rows that fall off the map
            box = ndimage.uniform_filter(d, size=k, mode=('constant', 'wrap')) * k * k
            cost[sy + SEARCH, sx + SEARCH] = box[np.ix_(cy, cx)]
    # featureless blocks (clear sky / uniform deck) carry no motion information
    m = ndimage.uniform_filter(a, size=k, mode=('nearest', 'wrap'))
    m2 = ndimage.uniform_filter(a * a, size=k, mode=('nearest', 'wrap'))
    sd = np.sqrt(np.maximum(m2 - m * m, 0))[np.ix_(cy, cx)]
    lat = 90 - (cy + 0.5) * 180 / H
    ok = (sd >= 0.045) & (np.abs(lat)[:, None] <= 66)
    flat = cost.reshape(S * S, GH, GW)
    best = flat.argmin(0)
    by, bx = best // S - SEARCH, best % S - SEARCH
    zero = cost[SEARCH, SEARCH]
    bestv = flat.min(0)
    noisy = (bestv > zero * 0.97) & ((bx != 0) | (by != 0))
    bx = np.where(noisy, 0, bx); by = np.where(noisy, 0, by)
    def at(dy, dx):
        yy = np.clip(by + dy + SEARCH, 0, S - 1); xx = np.clip(bx + dx + SEARCH, 0, S - 1)
        return np.take_along_axis(np.take_along_axis(cost, yy[None, None], 0)[0], xx[None], 0)[0]
    def para(l, c, r):
        d = l - 2 * c + r
        return np.where(d > 1e-6, np.clip(0.5 * (l - r) / np.where(d > 1e-6, d, 1), -0.5, 0.5), 0)
    c0 = at(0, 0)
    fx = bx + np.where(np.abs(bx) < SEARCH, para(at(0, -1), c0, at(0, 1)), 0)
    fy = by + np.where(np.abs(by) < SEARCH, para(at(-1, 0), c0, at(1, 0)), 0)
    fx = np.where(ok, fx, 0).astype(np.float32); fy = np.where(ok, fy, 0).astype(np.float32)
    fx = _median3(fx, ok); fy = _median3(fy, ok)
    fx, fy = _fill(fx, fy, ok)
    fx = _blur(fx); fy = _blur(fy)
    dpp = 360.0 / W
    east, north = fx * dpp / hours, -fy * dpp / hours
    enc = lambda v: np.clip(np.round(128 + v / FLOW_MAX * 127), 1, 255).astype(np.uint8)
    return np.stack([enc(east), enc(north)], -1)              # (32, 64, 2)

def still():
    return np.full((GH, GW, 2), 128, np.uint8)

def stats(f):
    v = np.hypot((f[..., 0].astype(float) - 128) / 127 * FLOW_MAX, (f[..., 1].astype(float) - 128) / 127 * FLOW_MAX)
    return float(v.mean()), float(np.percentile(v, 90)), float(v.max())

def _median3(f, ok):
    src = f.copy(); out = f.copy()
    for y in range(GH):
        for x in range(GW):
            if not ok[y, x]: continue
            w = [src[yy, (x + dx) % GW] for yy in range(max(0, y - 1), min(GH, y + 2)) for dx in (-1, 0, 1) if ok[yy, (x + dx) % GW]]
            out[y, x] = np.median(w)
    return out

def _fill(fx, fy, ok):
    have = ok.copy()
    for _ in range(24):
        nx, ny, nh = fx.copy(), fy.copy(), have.copy()
        for y in range(GH):
            for x in range(GW):
                if have[y, x]: continue
                sx = sy = n = 0
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    yy = y + dy
                    if yy < 0 or yy >= GH: continue
                    xx = (x + dx) % GW
                    if have[yy, xx]: sx += fx[yy, xx]; sy += fy[yy, xx]; n += 1
                if n: nx[y, x] = 0.85 * sx / n; ny[y, x] = 0.85 * sy / n; nh[y, x] = True
        fx, fy, have = nx, ny, nh
    return fx, fy

def _blur(f):
    k = np.array([0.25, 0.5, 0.25], np.float32)
    t = sum(k[d + 1] * np.roll(f, -d, axis=1) for d in (-1, 0, 1))
    pad = np.pad(t, ((1, 1), (0, 0)), mode='edge')
    return k[0] * pad[:-2] + k[1] * pad[1:-1] + k[2] * pad[2:]
