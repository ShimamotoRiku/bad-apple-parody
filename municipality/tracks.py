"""複数物体の場面に対応する版: 連結成分ごとに市区町村を割り当てる。

extract : 各フレームの図形を連結成分に分け、前フレームの成分と重なりで対応づけて「トラック」を作る。
          各成分を正規化し、全市区町村 × 反転（角度は最大）との IoU を計算する。
          画面外は「どちらでもない」として IoU から除外する（画面端で切れた図形の直線に引きずられないように）。
plan    : トラックごとに切り替えペナルティ lam 付きの Viterbi で市区町村を選び、区間ごとに角度を追従させる。
render  : プレビュー動画（音声なし）。

使い方:
  python tracks.py extract
  python tracks.py plan 1.0
  python tracks.py render 1.0 [開始] [終了]
"""
import os, sys, json, time, pickle
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
from common import BASE, G, ANG, WORDS, N_FRAMES, FRAME_DIR, load_munis, canon_points, raster_canon, pack, load_frame, canon_region
from score_all import load_bank

FW, FH = 240, 180
MIN_AREA = 0.002 * FW * FH       # これより小さい成分（花びら・粉など）は無視
LINK = 0.3                        # 重なり / 小さい方の面積 がこれ以上なら同じトラック
COMP = os.path.join(BASE, 'data', 'components.npz')
FPS = 30


def components(fig):
    n, lab, st, _ = cv2.connectedComponentsWithStats(fig.astype(np.uint8), connectivity=8)
    keep = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] >= MIN_AREA]
    return lab, keep


def score_comp(mask, bits, cnt, valid_full):
    T, V, tf = canon_region(mask, valid_full)
    if T is None or T.sum() == 0:
        return None
    t, v = pack(T), pack(V)
    inter = np.bitwise_count(bits & t).sum(-1)
    union = np.bitwise_count((bits | t) & v).sum(-1)
    iou = (inter / np.maximum(union, 1)).max(-1)          # 角度について最大 → (N, 2)
    return iou.astype(np.float16), t, v, np.array(tf)


def extract():
    bits, cnt = load_bank()
    valid_full = np.ones((FH, FW), bool)
    t0 = time.time()

    def per_frame(n):
        fig, inv = load_frame(n)
        lab, keep = components(fig)
        return n, inv, lab, keep

    with ThreadPoolExecutor(8) as ex:
        frames = list(ex.map(per_frame, range(1, N_FRAMES + 1)))
    print(f'成分抽出 {time.time() - t0:.0f}s', flush=True)

    # トラックの対応づけ（逐次）
    recs = []          # (frame, track, area, label) 各成分
    prev_lab, prev_keep, prev_tr = None, [], {}
    next_tr = 0
    for n, inv, lab, keep in frames:
        cur_tr = {}
        areas = {k: int((lab == k).sum()) for k in keep}
        if prev_lab is not None and keep and prev_keep:
            pairs = []
            for k in keep:
                ov = np.bincount(prev_lab[lab == k], minlength=prev_lab.max() + 1)
                for j in prev_keep:
                    if ov[j] > 0:
                        pairs.append((ov[j] / min(areas[k], prev_area[j]), k, j))
            used = set()
            for r, k, j in sorted(pairs, reverse=True):
                if r < LINK or k in cur_tr or j in used:
                    continue
                cur_tr[k] = prev_tr[j]; used.add(j)
        for k in keep:
            if k not in cur_tr:
                cur_tr[k] = next_tr; next_tr += 1
            recs.append((n, cur_tr[k], areas[k], k))
        prev_lab, prev_keep, prev_tr, prev_area = lab, keep, cur_tr, areas
    print(f'成分 {len(recs)} 個, トラック {next_tr} 本', flush=True)

    labs = {n: lab for n, _, lab, _ in frames}
    invs = np.array([inv for _, inv, _, _ in frames])

    def per_comp(r):
        n, tr, area, k = r
        return score_comp(labs[n] == k, bits, cnt, valid_full)

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        sc = list(ex.map(per_comp, recs))
    ok = [i for i, s in enumerate(sc) if s is not None]
    print(f'照合 {time.time() - t0:.0f}s', flush=True)
    np.savez(COMP, frame=np.array([recs[i][0] for i in ok]), track=np.array([recs[i][1] for i in ok]),
             area=np.array([recs[i][2] for i in ok]), iou=np.stack([sc[i][0] for i in ok]),
             tbits=np.stack([sc[i][1] for i in ok]), vbits=np.stack([sc[i][2] for i in ok]),
             tf=np.stack([sc[i][3] for i in ok]), inv=invs)
    print('saved', COMP)


def viterbi(S, lam):
    T, K = S.shape
    V = S[0].copy(); stay = np.zeros((T, K), bool); pb = np.zeros(T, np.int64)
    for t in range(1, T):
        j = int(V.argmax()); sw = V[j] - lam
        stay[t] = V >= sw; pb[t] = j
        V = np.where(stay[t], V, sw) + S[t]
    p = np.zeros(T, np.int64); p[-1] = V.argmax()
    for t in range(T - 1, 0, -1):
        p[t - 1] = p[t] if stay[t, p[t]] else pb[t]
    return p


def angle_track(polys, tb, vb, mir, mu=0.01):
    pts = [canon_points(polys, a, mir) for a in ANG]
    clipped = np.array([np.vstack(q).min() < 0 or np.vstack(q).max() > G for q in pts])   # グリッドからはみ出す角度は使わない
    bank = np.stack([pack(raster_canon(q)) for q in pts])
    A = len(ANG)
    inter = np.bitwise_count(bank[None] & tb[:, None]).sum(-1)
    union = np.bitwise_count((bank[None] | tb[:, None]) & vb[:, None]).sum(-1)
    sc = inter / np.maximum(union, 1)                       # (L, A)
    sc[:, clipped] = -1
    d = np.abs(np.arange(A)[:, None] - np.arange(A)[None]); d = np.minimum(d, A - d)
    V = sc[0].copy(); bp = np.zeros(sc.shape, np.int64)
    for t in range(1, len(sc)):
        M = V[:, None] - mu * d
        bp[t] = M.argmax(0); V = M.max(0) + sc[t]
    a = np.zeros(len(sc), np.int64); a[-1] = V.argmax()
    for t in range(len(sc) - 1, 0, -1):
        a[t - 1] = bp[t, a[t]]
    return a, sc[np.arange(len(sc)), a]


def plan(lam):
    z = np.load(COMP)
    names, polys, _ = load_munis()
    iou = np.nan_to_num(z['iou'].astype(np.float32)).reshape(len(z['frame']), -1)
    out = {}      # 成分 index → (muni, mir, ang, iou)
    nseg = 0; seglen = []
    for tr in np.unique(z['track']):
        idx = np.flatnonzero(z['track'] == tr)           # フレーム順に並んでいる
        p = viterbi(iou[idx], lam)
        cut = np.r_[0, np.flatnonzero(np.diff(p)) + 1, len(p)]
        for s, e in zip(cut[:-1], cut[1:]):
            i, mir = divmod(int(p[s]), 2)
            a, v = angle_track(polys[i], z['tbits'][idx[s:e]], z['vbits'][idx[s:e]], mir)
            for k, c in enumerate(idx[s:e]):
                out[int(c)] = (i, mir, int(a[k]), float(v[k]))
            nseg += 1; seglen.append(e - s)
    area = z['area']
    v = np.array([out[c][3] for c in range(len(area))])
    print(f'lam={lam}: 区間 {nseg}, 平均長 {np.mean(seglen) / FPS:.2f} 秒, IoU 平均 {v.mean():.3f}, '
          f'面積加重 {np.average(v, weights=area):.3f}')
    path = os.path.join(BASE, 'data', f'tplan_{lam}.json')
    json.dump([out[c] for c in range(len(area))], open(path, 'w'))
    print('saved', path)


def render(lam, lo, hi):
    from render import font, japan_map, OW, OH, BAR
    z = np.load(COMP)
    pl = json.load(open(os.path.join(BASE, 'data', f'tplan_{lam}.json')))
    names, polys, _ = load_munis()
    geoms = pickle.load(open(os.path.join(BASE, 'data', 'muni.pkl'), 'rb'))['geoms']
    base_map, xy = japan_map(geoms)
    f_name, f_small = font(22), font(15)
    by_frame = {}
    for c, n in enumerate(z['frame']):
        by_frame.setdefault(int(n), []).append(c)
    out = os.path.join(BASE, 'out', f'tpreview_{lam}_{lo:04d}_{hi:04d}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (OW * 2, OH + BAR))
    if not vw.isOpened():
        raise RuntimeError('VideoWriter を開けない: ' + out)
    cache = {}
    for n in range(lo, hi + 1):
        inv = bool(z['inv'][n - 1])
        fg, bg = ((0, 0, 0), (255, 255, 255)) if inv else ((255, 255, 255), (0, 0, 0))
        raw = Image.open(os.path.join(FRAME_DIR, f'frame-{n:04d}.jpg')).convert('RGB').resize((OW, OH), Image.BILINEAR)
        right = Image.new('RGB', (OW, OH), bg); dr = ImageDraw.Draw(right)
        cs = sorted(by_frame.get(n, []), key=lambda c: -z['area'][c])
        for c in cs:
            i, mir, a, v = pl[c]
            if (i, mir, a) not in cache:
                cache[(i, mir, a)] = canon_points(polys[i], ANG[a], mir)
            s, cx, cy = z['tf'][c]
            for q in cache[(i, mir, a)]:
                dr.polygon([tuple(p) for p in ((q - G / 2) / s + [cx, cy]) * (OW / FW)], fill=fg)
        canvas = Image.new('RGB', (OW * 2, OH + BAR), (20, 20, 20))
        canvas.paste(raw, (0, 0)); canvas.paste(right, (OW, 0))
        dc = ImageDraw.Draw(canvas)
        if cs:
            mp = base_map.copy(); dm = ImageDraw.Draw(mp)
            for rank, c in enumerate(cs):
                g = geoms[pl[c][0]]
                for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
                    a_ = np.array(p.exterior.coords); q = xy(a_[:, 0], a_[:, 1])
                    dm.polygon([tuple(v) for v in q], fill=(230, 60, 60) if rank == 0 else (240, 170, 60))
            g = geoms[pl[cs[0]][0]]; cc = xy(np.array([g.centroid.x]), np.array([g.centroid.y]))[0]
            dm.ellipse([cc[0] - 6, cc[1] - 6, cc[0] + 6, cc[1] + 6], outline=(255, 80, 80), width=2)
            canvas.paste(mp.resize((150, 150)), (OW * 2 - 160, OH - 160))
            label = '・'.join(names[pl[c][0]] for c in cs[:3]) + (f' ほか{len(cs) - 3}' if len(cs) > 3 else '')
            dc.text((12, OH + 12), label, font=f_name if len(cs) == 1 else f_small, fill=(255, 255, 255))
            dc.text((OW * 2 - 250, OH + 18), f'frame {n}  IoU {pl[cs[0]][3]:.2f}', font=f_small, fill=(180, 180, 180))
        vw.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
    vw.release()
    print('saved', out)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'extract':
        extract()
    elif cmd == 'plan':
        plan(float(sys.argv[2]))
    elif cmd == 'render':
        render(float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1,
               int(sys.argv[4]) if len(sys.argv) > 4 else N_FRAMES)
    else:
        print(__doc__); sys.exit(1)
