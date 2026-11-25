import numpy as np
from PIL import Image, ImageDraw
import sys, time

W, H = 64, 48
rng = np.random.default_rng(0)

def target_apple():
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    d.ellipse([18, 12, 34, 42], fill=255); d.ellipse([30, 12, 46, 42], fill=255)
    d.rectangle([31, 4, 33, 14], fill=255); d.ellipse([34, 4, 42, 10], fill=255)
    return np.array(im) > 127

def target_person():
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    d.ellipse([27, 3, 37, 13], fill=255)                 # head
    d.polygon([(26, 14), (38, 14), (40, 30), (24, 30)], fill=255)  # torso
    d.polygon([(26, 14), (16, 6), (14, 8), (25, 18)], fill=255)    # raised arm
    d.polygon([(38, 14), (50, 22), (48, 24), (38, 19)], fill=255)  # arm
    d.polygon([(25, 30), (31, 30), (24, 46), (19, 46)], fill=255)  # leg
    d.polygon([(33, 30), (39, 30), (46, 44), (41, 46)], fill=255)  # leg
    return np.array(im) > 127

def escape(cx, cy, scale, rot, maxit=400):
    # batch of windows -> smooth iteration counts (N,H,W); inside = maxit
    xs = (np.arange(W) - W / 2) / W
    ys = (np.arange(H) - H / 2) / W
    X, Y = np.meshgrid(xs, ys)
    N = len(cx)
    c, s = np.cos(rot)[:, None, None], np.sin(rot)[:, None, None]
    sc = scale[:, None, None]
    re = cx[:, None, None] + sc * (c * X - s * Y)
    im = cy[:, None, None] + sc * (s * X + c * Y)
    C = re + 1j * im
    Z = np.zeros_like(C); out = np.full(C.shape, float(maxit)); alive = np.ones(C.shape, bool)
    for i in range(maxit):
        Z[alive] = Z[alive] ** 2 + C[alive]
        esc = alive & (np.abs(Z) > 4)
        if esc.any():
            out[esc] = i + 1 - np.log2(np.log(np.abs(Z[esc])))
            alive &= ~esc
        if not alive.any(): break
    return out

def best_score(F, T):
    # choose threshold (and polarity) maximizing mean IoU of both classes
    best = (-1, None, None)
    flat = F.reshape(F.shape[0], -1)
    qs = np.quantile(flat, np.linspace(0.05, 0.95, 37), axis=1).T  # (N,37)
    t = T.ravel()
    res = []
    for n in range(F.shape[0]):
        bs = (-1, 0, 0)
        for th in qs[n]:
            for pol in (0, 1):
                B = (flat[n] >= th) if pol == 0 else (flat[n] < th)
                i1 = (B & t).sum() / max((B | t).sum(), 1)
                i0 = (~B & ~t).sum() / max((~B | ~t).sum(), 1)
                sc = (i1 + i0) / 2
                if sc > bs[0]: bs = (sc, th, pol)
        res.append(bs)
    return res

# boundary points for sampling centers
g = 800
xs = np.linspace(-2.1, 0.6, g); ys = np.linspace(-1.2, 1.2, g)
X, Y = np.meshgrid(xs, ys); C = X + 1j * Y; Z = np.zeros_like(C); inside = np.ones(C.shape, bool)
for _ in range(200):
    Z[inside] = Z[inside] ** 2 + C[inside]; inside &= np.abs(Z) <= 2
bd = inside ^ np.roll(inside, 1, 0) | inside ^ np.roll(inside, 1, 1)
by, bx = np.nonzero(bd)
pts = C[by, bx]

def run(T, name, n_random=6000, batch=300, n_refine=8, refine_steps=300):
    t0 = time.time()
    cands = []
    for b in range(n_random // batch):
        k = rng.integers(0, len(pts), batch)
        cx = pts[k].real + rng.normal(0, 0.002, batch); cy = pts[k].imag + rng.normal(0, 0.002, batch)
        scale = 10 ** rng.uniform(-7, -0.5, batch); rot = rng.uniform(0, 2 * np.pi, batch)
        F = escape(cx, cy, scale, rot)
        for j, (sc, th, pol) in enumerate(best_score(F, T)):
            cands.append((sc, cx[j], cy[j], scale[j], rot[j]))
    cands.sort(key=lambda r: -r[0])
    print(name, "random best", round(cands[0][0], 3), "time", round(time.time() - t0, 1), flush=True)
    # hill-climb top few
    best_all = None
    for sc0, cx, cy, s, r in cands[:n_refine]:
        cur = (sc0, cx, cy, s, r)
        for step in range(refine_steps // 20):
            m = 20
            ncx = cur[1] + rng.normal(0, 0.05, m) * cur[3]
            ncy = cur[2] + rng.normal(0, 0.05, m) * cur[3]
            ns = cur[3] * np.exp(rng.normal(0, 0.08, m)); nr = cur[4] + rng.normal(0, 0.08, m)
            F = escape(ncx, ncy, ns, nr)
            for j, (sc, th, pol) in enumerate(best_score(F, T)):
                if sc > cur[0]: cur = (sc, ncx[j], ncy[j], ns[j], nr[j])
        if best_all is None or cur[0] > best_all[0]: best_all = cur
    print(name, "refined", round(best_all[0], 3), best_all[1:], flush=True)
    sc, cx, cy, s, r = best_all
    F = escape(np.array([cx]), np.array([cy]), np.array([s]), np.array([r]))
    (_, th, pol), = best_score(F, T)
    B = (F[0] >= th) if pol == 0 else (F[0] < th)
    # render: target | binarized match | raw field (log)
    raw = np.log1p(F[0]); raw = (raw - raw.min()) / (np.ptp(raw) + 1e-9)
    row = np.concatenate([T * 1.0, np.ones((H, 2)) * 0.5, B * 1.0, np.ones((H, 2)) * 0.5, raw], 1)
    Image.fromarray((row * 255).astype(np.uint8)).resize((row.shape[1] * 6, H * 6), Image.NEAREST).save(f"{name}.png")
    return best_all

run(target_apple(), "apple")
run(target_person(), "person")
