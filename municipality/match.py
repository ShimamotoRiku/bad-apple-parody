# 市区町村（とその隣接結合）からシルエットに合う形を探す
import pickle, numpy as np, cv2, time, glob
from PIL import Image, ImageDraw, ImageFont
from shapely.ops import unary_union

import os
BASE = os.path.dirname(os.path.abspath(__file__))
W, H = int(os.environ.get('MW', 64)), int(os.environ.get('MH', 48))
D = pickle.load(open(os.path.join(BASE, 'data/muni.pkl'), 'rb'))
names, geoms, adj = D['names'], D['geoms'], D['adj']
N = len(names)

def parts(g):
    ps = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    amax = max(p.area for p in ps)
    return [np.array(p.exterior.coords) for p in ps if p.area > 0.05 * amax]  # 小さい離島は除外

P = [parts(g) for g in geoms]
LAT = np.array([g.centroid.y for g in geoms])

ANG = np.radians(np.arange(0, 360, 7.5))

def raster(polys, T, ang, mir, sfac=1.0, dx=0.0, dy=0.0):
    ys, xs = np.nonzero(T)
    tcx, tcy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    tw, th = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    c, s = np.cos(ang), np.sin(ang)
    tr = []
    for a in polys:
        x, y = a[:, 0], -a[:, 1]          # 北を上に
        if mir: x = -x
        tr.append(np.c_[c * x - s * y, s * x + c * y])
    allp = np.vstack(tr)
    mn, mx = allp.min(0), allp.max(0)
    w, h = mx - mn
    sc = np.sqrt(tw * th / max(w * h, 1e-12)) * sfac
    ctr = (mn + mx) / 2
    img = np.zeros((H, W), np.uint8)
    pts = [np.round(((q - ctr) * sc + [tcx + dx, tcy + dy]) * 16).astype(np.int32) for q in tr]
    cv2.fillPoly(img, pts, 1, lineType=cv2.LINE_8, shift=4)
    return img.astype(bool)

def iou(B, T):
    return (B & T).sum() / max((B | T).sum(), 1)

def polys_of(members):
    k = np.cos(np.radians(np.mean(LAT[list(members)])))
    return [np.c_[a[:, 0] * k, a[:, 1]] for m in members for a in P[m]]

def best_pose(polys, T, angs=ANG):
    best = (-1, 0, 0)
    for ang in angs:
        for mir in (0, 1):
            v = iou(raster(polys, T, ang, mir), T)
            if v > best[0]: best = (v, ang, mir)
    return best

def refine(polys, T, pose):
    v, ang, mir = pose; sf, dx, dy = 1.0, 0.0, 0.0
    rng = np.random.default_rng(1)
    for _ in range(150):
        cand = (ang + rng.normal(0, 0.05), sf * np.exp(rng.normal(0, 0.05)), dx + rng.normal(0, 0.7), dy + rng.normal(0, 0.7))
        nv = iou(raster(polys, T, cand[0], mir, cand[1], cand[2], cand[3]), T)
        if nv > v: v, (ang, sf, dx, dy) = nv, cand
    return v, (ang, mir, sf, dx, dy)

def search(T, beam=40, max_size=6):
    t0 = time.time()
    singles = []
    for i in range(N):
        v, a, m = best_pose(polys_of([i]), T)
        singles.append((v, frozenset([i])))
    singles.sort(key=lambda r: -r[0])
    print(f'  singles done {time.time()-t0:.0f}s best {singles[0][0]:.3f} {names[next(iter(singles[0][1]))]}', flush=True)
    best_by_size = {1: singles[0]}
    frontier = singles[:beam]
    seen = set(s for _, s in frontier)
    for size in range(2, max_size + 1):
        cands = []
        for _, S in frontier:
            nb = set().union(*(adj[m] for m in S)) - S
            for j in nb:
                S2 = S | {j}
                if S2 in seen: continue
                seen.add(S2)
                v, a, m = best_pose(polys_of(S2), T, ANG[::2])  # 粗い角度で
                cands.append((v, S2))
        cands.sort(key=lambda r: -r[0])
        frontier = cands[:beam]
        best_by_size[size] = frontier[0]
        print(f'  size {size}: {frontier[0][0]:.3f} ({len(cands)} evals, {time.time()-t0:.0f}s)', flush=True)
    return singles, best_by_size

font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 20, index=0)

def render(T, members, pose, path, label):
    v, (ang, mir, sf, dx, dy) = pose
    S = 8
    big = np.kron(T, np.ones((S, S), bool))
    # 結果を高解像度で描く（目標と同じ変換を S 倍で）
    polys = polys_of(members)
    Tb = big
    img = Image.new('RGB', (W * S * 2 + 20, H * S + 60), (40, 40, 40))
    img.paste(Image.fromarray((big * 255).astype(np.uint8)).convert('RGB'), (0, 0))
    canvas = Image.new('RGB', (W * S, H * S), (0, 0, 0)); dr = ImageDraw.Draw(canvas)
    ys, xs = np.nonzero(T)
    tcx, tcy = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    tw, th = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    c, s = np.cos(ang), np.sin(ang)
    tr_each = []
    for m in members:
        k = np.cos(np.radians(np.mean(LAT[list(members)])))
        tr_each.append([])
        for a in P[m]:
            x, y = a[:, 0] * k, -a[:, 1]
            if mir: x = -x
            tr_each[-1].append(np.c_[c * x - s * y, s * x + c * y])
    allp = np.vstack([q for L in tr_each for q in L]); mn, mx = allp.min(0), allp.max(0)
    w, h = mx - mn; sc = np.sqrt(tw * th / (w * h)) * sf; ctr = (mn + mx) / 2
    for L in tr_each:
        for q in L:
            pp = ((q - ctr) * sc + [tcx + dx, tcy + dy]) * S
            dr.polygon([tuple(p) for p in pp], fill=(255, 255, 255))
    for L in tr_each:
        for q in L:
            pp = ((q - ctr) * sc + [tcx + dx, tcy + dy]) * S
            dr.line([tuple(p) for p in pp] + [tuple(pp[0])], fill=(150, 150, 150), width=1)
    img.paste(canvas, (W * S + 20, 0))
    ImageDraw.Draw(img).text((8, H * S + 14), f'{label}  IoU={v:.2f}', font=font, fill=(255, 255, 255))
    img.save(path)

if __name__ == '__main__':
    import sys
    os.makedirs(os.path.join(BASE, 'out'), exist_ok=True)
    tdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'targets')
    for path in sorted(glob.glob(tdir + '/*.png')):
        nm = path.split('/')[-1][:-4]
        T = np.array(Image.open(path).convert('L').resize((W, H), Image.LANCZOS)) > 127
        print(nm, flush=True)
        singles, bbs = search(T)
        for tag, (v, S) in [('single', singles[0])] + [(f'best', max(bbs.values(), key=lambda r: r[0]))]:
            pose = refine(polys_of(S), T, best_pose(polys_of(S), T))
            label = '＋'.join(names[m] for m in sorted(S, key=lambda m: names[m]))
            print(f'  {tag}: IoU {pose[0]:.3f}  {label}', flush=True)
            render(T, S, pose, os.path.join(BASE, 'out', f'{nm}_{tag}.png'), label)
