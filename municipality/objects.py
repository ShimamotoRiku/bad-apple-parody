"""ちらつき対策 改良版（tracks.py の発展）。

① 追跡を切れにくくする
  - 近い塊（腕と体など）は膨張処理でまとめて 1 物体として扱う。照合もまとめた形で行う。
  - 分裂・合体などで新しいトラックが始まるとき、重なっていた親トラックの市区町村を引き継ぐと得になるようにする。
② 切り替えのタイミング
  - 場面転換（カット）を検出し、カットでは切り替えの減点をほぼ 0、それ以外では強くする。
  - 最低保持時間 MIN_HOLD フレーム（カット以外）を Viterbi の制約として入れる。
③ 見え方（render）
  - 位置・大きさを時間方向に平滑化。
  - カット以外の切り替えは、前後数フレームで符号付き距離場を混ぜてモーフィングする。
  - 名前はフェードで切り替え、短い区間では出さない。

使い方:
  python objects.py extract
  python objects.py plan [lam] [--no-mirror]   # 既定 lam=3。--no-mirror で鏡像を使わない
  python objects.py render [開始] [終了]
"""
import os, sys, json, time, pickle
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
from common import BASE, G, ANG, N_FRAMES, FRAME_DIR, load_munis, load_geoms, canon_points
from score_all import load_bank
from tracks import score_comp, angle_track

FW, FH = 240, 180
FPS = 30
MIN_AREA = 0.002 * FW * FH
GROUP_R = 4                      # まとめる距離（240 幅で 4px ≒ 画面幅の 1.7%）
LINK = 0.3
CUT_XOR, CUT_DIFF = 0.08, 0.7    # カット判定: 画面の変化率 > CUT_XOR かつ 図形の (1 - IoU) > CUT_DIFF、または白黒反転
LAM_CUT = 0.05
MIN_HOLD = 15                    # 0.5 秒
AREA_BIAS = 0.0                  # 都道府県・地方の得点から引く値
AREA_MIN_SEC, AREA_MIN_IOU = 0.5, 0.6   # 都道府県・地方を残す区間の条件（長さ・平均一致度）
OBJ = os.path.join(BASE, 'data', 'objects.npz')
FIGS = os.path.join(BASE, 'data', 'figs.npz')


def load_figs():
    if not os.path.exists(FIGS):
        from common import load_frame
        with ThreadPoolExecutor(8) as ex:
            F = list(ex.map(load_frame, range(1, N_FRAMES + 1)))
        np.savez_compressed(FIGS, figs=np.packbits(np.stack([f for f, _ in F]), axis=-1), inv=np.array([i for _, i in F]))
    z = np.load(FIGS)
    return np.unpackbits(z['figs'], axis=-1)[..., :FW].astype(bool), z['inv']


def detect_cuts(figs, inv):
    a, b = figs[1:], figs[:-1]
    x = (a ^ b).mean((1, 2))
    d = 1 - (a & b).sum((1, 2)) / np.maximum((a | b).sum((1, 2)), 1)
    cut = np.zeros(len(figs), bool)                      # cut[t]: フレーム t（0 始まり）の直前で場面が変わる
    cut[1:] = (inv[1:] != inv[:-1]) | ((x > CUT_XOR) & (d > CUT_DIFF))
    cut[0] = True
    return cut


KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROUP_R + 1, 2 * GROUP_R + 1))


def objects_of(fig):
    """膨張した図形の連結成分を 1 物体とし、物体ごとの（元の図形の）画素数を返す"""
    n, lab = cv2.connectedComponents(cv2.dilate(fig.astype(np.uint8), KER), connectivity=8)
    area = np.bincount(lab[fig], minlength=n)
    return lab, [k for k in range(1, n) if area[k] >= MIN_AREA], area


def extract():
    figs, inv = load_figs()
    cut = detect_cuts(figs, inv)
    bits, cnt = load_bank()
    valid_full = np.ones((FH, FW), bool)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        objs = list(ex.map(objects_of, figs))
    recs = []            # (t, track, area, label)
    parent = {}          # track → 親 track（引き継ぎ元）
    prev = None; next_tr = 0
    for t, (lab, keep, area) in enumerate(objs):
        cur = {}
        if prev is not None and keep and prev[1] and not cut[t]:
            plab, pkeep, parea, ptr = prev
            pairs = []
            ov_of = {}
            for k in keep:
                ov = np.bincount(plab[(lab == k) & figs[t]], minlength=plab.max() + 1)
                ov_of[k] = ov
                for j in pkeep:
                    if ov[j] > 0:
                        pairs.append((ov[j] / min(area[k], parea[j]), k, j))
            used = set()
            for r, k, j in sorted(pairs, reverse=True):
                if r < LINK or k in cur or j in used:
                    continue
                cur[k] = ptr[j]; used.add(j)
            for k in keep:
                if k not in cur:
                    cur[k] = next_tr; next_tr += 1
                    ov = ov_of[k]
                    js = [j for j in pkeep if ov[j] > 0]
                    if js:
                        parent[cur[k]] = ptr[max(js, key=lambda j: ov[j])]
        for k in keep:
            if k not in cur:
                cur[k] = next_tr; next_tr += 1
            recs.append((t, cur[k], int(area[k]), k))
        prev = (lab, keep, area, cur)
    print(f'物体 {len(recs)} 個, トラック {next_tr} 本（うち親あり {len(parent)}）, カット {cut.sum()}', flush=True)

    def per(r):
        t, tr, a, k = r
        return score_comp((objs[t][0] == k) & figs[t], bits, cnt, valid_full)

    with ThreadPoolExecutor(8) as ex:
        sc = list(ex.map(per, recs))
    ok = [i for i, s in enumerate(sc) if s is not None]
    print(f'照合 {time.time() - t0:.0f}s', flush=True)
    np.savez(OBJ, frame=np.array([recs[i][0] + 1 for i in ok]), track=np.array([recs[i][1] for i in ok]),
             area=np.array([recs[i][2] for i in ok]), iou=np.stack([sc[i][0] for i in ok]),
             tbits=np.stack([sc[i][1] for i in ok]), vbits=np.stack([sc[i][2] for i in ok]),
             tf=np.stack([sc[i][3] for i in ok]), inv=inv, cut=cut,
             parent=np.array([[c, p] for c, p in parent.items()]).reshape(-1, 2))
    print('saved', OBJ)


def viterbi_hold(S, lam_t, D, bonus=None):
    """最低保持 D フレームつき Viterbi。状態 (k, d)、d = 切り替えてからの経過（D で頭打ち）。
    切り替えは d == D から（lam_t[t] が LAM_CUT 以下のカットでは任意の d から）可能。lam_t[t] はフレーム t に切り替える減点。"""
    T, K = S.shape
    NEG = -1e9
    V = np.full((K, D), NEG)          # 列 d-1
    V[:, 0] = S[0] + (bonus if bonus is not None else 0)
    src = np.zeros(T, np.int64); srcd = np.zeros(T, np.int64); keepD = np.zeros((T, K), bool)
    for t in range(1, T):
        if lam_t[t] <= LAM_CUT:
            flat = V.argmax(); j, dj = divmod(int(flat), D)
        else:
            j, dj = int(V[:, D - 1].argmax()), D - 1
        b = V[j, dj] - lam_t[t]
        src[t], srcd[t] = j, dj
        N = np.empty_like(V)
        N[:, 0] = b
        N[:, 1:D - 1] = V[:, 0:D - 2]
        if D > 1:
            keepD[t] = V[:, D - 1] >= V[:, D - 2] if D > 1 else True
            N[:, D - 1] = np.maximum(V[:, D - 1], V[:, D - 2])
        V = N + S[t][:, None]
    k, d = divmod(int(V.argmax()), D)
    path = np.zeros(T, np.int64); path[-1] = k
    for t in range(T - 1, 0, -1):
        if d == 0:
            k, d = src[t], srcd[t]
        elif d == D - 1:
            d = D - 1 if keepD[t, k] else D - 2
        else:
            d -= 1
        path[t - 1] = k
    return path


def plan(lam=3.0, mirror=True):
    z = np.load(OBJ)
    names, polys, _ = load_munis()
    N_MUNI = sum(1 for k in load_geoms()[2] if k == '市区町村')
    iou = np.nan_to_num(z['iou'].astype(np.float32))
    if not mirror:
        iou[:, :, 1] = 0          # 鏡像は実在しない形なので使わない（航空写真を貼るため）
    area_ix = np.array([k != '市区町村' for k in load_geoms()[2]])
    iou[:, area_ix] -= AREA_BIAS
    iou = iou.reshape(len(z['frame']), -1)
    cut = z['cut']
    parent = {int(c): int(p) for c, p in z['parent']}
    frame = z['frame']; track = z['track']
    order = sorted(np.unique(track), key=lambda tr: frame[track == tr].min())
    ban = {}              # トラック → 使わせない都道府県・地方（短すぎる・合っていない区間になったもの）
    for it in range(6):
        state_at = {}         # (track, frame) → 状態
        used = {}             # フレーム → 先に割り当てたトラックが使っている市区町村（同じフレームで同じ形を2回出さない）
        out = {}
        nsw = 0; nsw_cut = 0; new_ban = 0
        for tr in order:
            idx = np.flatnonzero(track == tr)
            fr = frame[idx]
            S = iou[idx].copy()
            for k, f in enumerate(fr):
                for m in used.get(int(f), ()):
                    S[k, 2 * m:2 * m + 2] = -100
            for m in ban.get(int(tr), ()):
                S[:, 2 * m:2 * m + 2] = -100
            lam_t = np.where(cut[fr - 1], LAM_CUT, lam)
            bonus = None
            p = parent.get(int(tr))
            if p is not None and (p, fr[0] - 1) in state_at:
                bonus = np.zeros(iou.shape[1]); bonus[state_at[(p, fr[0] - 1)]] = lam   # 親と同じ市区町村なら切り替え扱いにしない
            path = viterbi_hold(S, lam_t, MIN_HOLD, bonus)
            for f, s in zip(fr, path):
                state_at[(int(tr), int(f))] = int(s)
                used.setdefault(int(f), set()).add(int(s) // 2)
            cuts_ = np.r_[0, np.flatnonzero(np.diff(path)) + 1, len(path)]
            for s, e in zip(cuts_[:-1], cuts_[1:]):
                if s > 0:
                    nsw += 1; nsw_cut += bool(cut[fr[s] - 1])
                i, mir = divmod(int(path[s]), 2)
                a, v = angle_track(polys[i], z['tbits'][idx[s:e]], z['vbits'][idx[s:e]], mir)
                for k, c in enumerate(idx[s:e]):
                    out[int(c)] = (i, mir, int(a[k]), float(v[k]))
                # 都道府県・地方は AREA_MIN_SEC 秒以上続き、平均一致度 AREA_MIN_IOU 以上の区間だけ残す
                if i >= N_MUNI and (e - s < AREA_MIN_SEC * FPS or v.mean() < AREA_MIN_IOU):
                    ban.setdefault(int(tr), set()).add(i); new_ban += 1
        if not new_ban:
            break
        print(f'  {it + 1} 回目: 条件を満たさない都道府県・地方の区間 {new_ban} → 使わずに割り当て直す', flush=True)
    else:
        raise RuntimeError('都道府県・地方の割り当てが収束しない')
    v = np.array([out[c][3] for c in range(len(frame))])
    print(f'lam={lam}: トラック内の切り替え {nsw}（うちカット上 {nsw_cut}）, IoU 平均 {v.mean():.3f}, 面積加重 {np.average(v, weights=z["area"]):.3f}')
    report_main(frame, z['area'], [out[c][0] for c in range(len(frame))], cut)
    per = {}
    for c, f in enumerate(frame):
        per.setdefault(int(f), []).append(out[c][0])
    print(f'  同じフレームで同じ形が重なるフレーム: {sum(len(v) != len(set(v)) for v in per.values())}')
    n_area = sum(1 for c in range(len(frame)) if out[c][0] >= N_MUNI)
    print(f'  都道府県・地方を使った物体フレーム: {n_area} / {len(frame)}')
    path = os.path.join(BASE, 'data', f'oplan_{lam}{"" if mirror else "_nomir"}.json')
    json.dump([out[c] for c in range(len(frame))], open(path, 'w'))
    print('saved', path)


def report_main(frame, area, muni, cut):
    """見た目のちらつきの指標: 各フレームで最大の物体に当てた市区町村が、何秒ごとに変わるか（カット以外で）"""
    main = {}
    for c, n in enumerate(frame):
        if n not in main or area[c] > area[main[n]]:
            main[n] = c
    ns = sorted(main)
    ch = [(a, b) for a, b in zip(ns, ns[1:]) if muni[main[a]] != muni[main[b]]]
    ch_nc = [1 for a, b in ch if not cut[b - 1]]
    print(f'  最大物体の市区町村: {len(ch)} 回切り替え → 平均 {len(ns) / max(len(ch), 1) / FPS:.2f} 秒に1回'
          f'（カット以外での切り替え {len(ch_nc)} 回 → {len(ns) / max(len(ch_nc), 1) / FPS:.2f} 秒に1回）')


# ---------------- 描画 ----------------
OW, OH, BAR = 640, 480, 56
MORPH = 8                      # モーフィングのフレーム数（切り替えの前後 4 フレームずつ）
SMOOTH_SIGMA = 1.5             # 位置・大きさの平滑化（フレーム）
LABEL_MIN = 20                 # これより短い区間では名前を出さない
FADE = 6


def gsmooth(x, sigma):
    if len(x) < 3:
        return x
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, ((r, r),) + ((0, 0),) * (x.ndim - 1), mode='edge')
    return np.stack([np.convolve(xp[:, j], k, mode='valid') for j in range(x.shape[1])], 1)


def sdf(mask):
    m = mask.astype(np.uint8)
    return cv2.distanceTransform(1 - m, cv2.DIST_L2, 3) - cv2.distanceTransform(m, cv2.DIST_L2, 3)


def build_draw(z, pl):
    """位置・大きさの平滑化と、各物体の描画指示（モーフィング込み）、名前表示の区間を作る。
    戻り値: tf（平滑化済み）, draw {物体 → [(市区町村, 反転, 角度, 重み)]}, by_frame {フレーム → [物体]}, label_alpha {フレーム → (市区町村, 不透明度)}"""
    frame, track, area, cut = z['frame'], z['track'], z['area'], z['cut']
    tf = z['tf'].astype(float).copy()
    draw = {}             # c → [(muni, mir, ang, 重み)]  重みは SDF 混合用
    for tr in np.unique(track):
        idx = np.flatnonzero(track == tr); fr = frame[idx]
        pieces = np.split(np.arange(len(idx)), np.flatnonzero(cut[fr[1:] - 1]) + 1)
        for pc in pieces:
            x = np.c_[np.log(tf[idx[pc], 0]), tf[idx[pc], 1:]]
            x = gsmooth(x, SMOOTH_SIGMA)
            tf[idx[pc], 0] = np.exp(x[:, 0]); tf[idx[pc], 1:] = x[:, 1:]
        st = [(pl[c][0], pl[c][1]) for c in idx]
        for k, c in enumerate(idx):
            draw[int(c)] = [(pl[c][0], pl[c][1], pl[c][2], 1.0)]
        # カット以外での切り替えにモーフィングを入れる
        for k in range(1, len(idx)):
            if st[k] == st[k - 1] or cut[fr[k] - 1]:
                continue
            A = pl[idx[k - 1]]; B = pl[idx[k]]
            for m in range(-MORPH // 2, MORPH // 2):
                kk = k + m
                if kk < 0 or kk >= len(idx):
                    continue
                al = (m + MORPH // 2 + 0.5) / MORPH          # 0→1
                draw[int(idx[kk])] = [(A[0], A[1], A[2], 1 - al), (B[0], B[1], B[2], al)]

    # 名前表示: 最大物体の市区町村の区間を作る
    by_frame = {}
    for c, n in enumerate(frame):
        by_frame.setdefault(int(n), []).append(c)
    main_muni = {n: pl[max(cs, key=lambda c: area[c])][0] for n, cs in by_frame.items()}
    label_alpha = {}
    ns = list(range(1, N_FRAMES + 1)); runs = []; cur = None
    for n in ns:
        m = main_muni.get(n)
        if cur and cur[0] == m:
            cur[2] = n
        else:
            cur = [m, n, n]; runs.append(cur)
    for m, s, e in runs:
        if m is None or e - s + 1 < LABEL_MIN:
            continue
        for n in range(s, e + 1):
            label_alpha[n] = (m, min(1, (n - s + 1) / FADE, (e - n + 1) / FADE))
    return tf, draw, by_frame, label_alpha


def render(lam, lo, hi):
    from render import font, japan_map
    z = np.load(OBJ)
    pl = json.load(open(os.path.join(BASE, 'data', f'oplan_{lam}.json')))
    names, polys, _ = load_munis()
    geoms = pickle.load(open(os.path.join(BASE, 'data', 'muni.pkl'), 'rb'))['geoms']
    base_map, xy = japan_map(geoms)
    f_name, f_small = font(22), font(15)
    tf, draw, by_frame, label_alpha = build_draw(z, pl)

    out = os.path.join(BASE, 'out', f'opreview_{lam}_{lo:04d}_{hi:04d}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (OW * 2, OH + BAR))
    if not vw.isOpened():
        raise RuntimeError('VideoWriter を開けない: ' + out)
    cache = {}
    sc = OW / FW

    def shape_mask(i, mir, a, t):
        key = (i, mir, a)
        if key not in cache:
            cache[key] = canon_points(polys[i], ANG[a], mir)
        s, cx, cy = t
        img = np.zeros((OH, OW), np.uint8)
        pts = [np.round(((q - G / 2) / s + [cx, cy]) * sc * 16).astype(np.int32) for q in cache[key]]
        cv2.fillPoly(img, pts, 1, lineType=cv2.LINE_AA, shift=4)
        return img > 0

    map_cache = {}
    map_base = base_map.resize((150, 150)); map_box = (OW * 2 - 160, OH - 160)
    for n in range(lo, hi + 1):
        inv = bool(z['inv'][n - 1])
        fgv, bgv = (0, 255) if inv else (255, 0)
        raw = Image.open(os.path.join(FRAME_DIR, f'frame-{n:04d}.jpg')).convert('RGB').resize((OW, OH), Image.BILINEAR)
        R = np.zeros((OH, OW), bool)
        for c in by_frame.get(n, []):
            parts = draw[c]
            if len(parts) == 1:
                R |= shape_mask(*parts[0][:3], tf[c])
            else:
                acc = sum(w * sdf(shape_mask(i, mir, a, tf[c])) for i, mir, a, w in parts)
                R |= acc < 0
        right = Image.fromarray(np.where(R, fgv, bgv).astype(np.uint8)).convert('RGB')
        canvas = Image.new('RGB', (OW * 2, OH + BAR), (20, 20, 20))
        canvas.paste(raw, (0, 0)); canvas.paste(right, (OW, 0))
        canvas.paste(map_base, map_box)
        if n in label_alpha:
            m, al = label_alpha[n]
            if m not in map_cache:
                mp = base_map.copy(); dm = ImageDraw.Draw(mp); g = geoms[m]
                for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
                    q = np.array(p.exterior.coords); q = xy(q[:, 0], q[:, 1])
                    dm.polygon([tuple(v) for v in q], fill=(230, 60, 60))
                cc = xy(np.array([g.centroid.x]), np.array([g.centroid.y]))[0]
                dm.ellipse([cc[0] - 6, cc[1] - 6, cc[0] + 6, cc[1] + 6], outline=(255, 80, 80), width=2)
                map_cache[m] = mp.resize((150, 150))
            canvas.paste(Image.blend(map_base, map_cache[m], al), map_box)   # 下地は常に出し、強調だけフェード
            dc = ImageDraw.Draw(canvas)
            v = int(255 * al)
            dc.text((12, OH + 12), names[m], font=f_name, fill=(v, v, v))
        vw.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
    vw.release()
    print('saved', out)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'extract':
        extract()
    elif cmd == 'plan':
        plan(float(sys.argv[2]) if len(sys.argv) > 2 else 3.0, mirror='--no-mirror' not in sys.argv)
    elif cmd == 'render':
        render(float(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 1,
               int(sys.argv[4]) if len(sys.argv) > 4 else N_FRAMES)
    else:
        print(__doc__); sys.exit(1)
