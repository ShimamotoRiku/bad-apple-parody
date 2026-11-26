"""案C の全編動画（音声なし、1280x720）。

主画面（左 960x720）: 各物体の形を、その市区町村の航空写真（モノクロ）で塗る。
  明るさは図形の色に寄せる（白い図形 → 明るい側、黒い図形 → 暗い側）。
  明るさの正規化は市区町村ごとに 1 回だけ決める（フレームごとに決めると明滅するため）。
  切り替え時のモーフィングは、形は符号付き距離場の混合、写真は両方の重み付き平均。
右の帯: 元の映像（上端、見出しなし）と、映っている市区町村（最大 3）ごとの区画（県名・市区町村名、カラー航空写真（北が上）、日本地図上の位置）。
  その瞬間に主画面にある形だけを出す（0.27 秒以上続く形と、最大の形）。組が変わるときは前の組を消してから次を出す。
最後に出典を表示する。

使い方: python video_c.py [開始フレーム] [終了フレーム]
出力: out/video_c_<開始>_<終了>.mp4
"""
import os, sys, json, pickle, time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from common import BASE, ANG, N_FRAMES, FRAME_DIR, load_munis, load_geoms
from objects import OBJ, FPS, FADE, build_draw, sdf
from design_proto import h3, pose_affine, fit_affine, MW
from render import font, japan_map
from aerial import mosaic, local_to_px_affine, PHOTO_CREDIT, LANDSAT_CREDIT

W, H = 1280, 720
SW = W - MW
PLAN = os.path.join(BASE, 'data', 'oplan_3.0_nomir.json')
TONE = {False: (150.0, 255.0), True: (0.0, 100.0)}     # inv=False（白い図形）/ True（黒い図形）の明るさの範囲
CREDIT_SEC = 5


class Aerials:
    def __init__(self, names, polys, geoms):
        self.names, self.polys, self.geoms = names, polys, geoms
        self.d = {}
        self.zooms = set()

    def get(self, i):
        if i not in self.d:
            img, to_px, z = mosaic(self.geoms[i])
            Mpx = local_to_px_affine(self.geoms[i], to_px)
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask = np.zeros(g.shape, np.uint8)
            pts = [np.round((h3(Mpx) @ np.c_[q, np.ones(len(q))].T)[:2].T).astype(np.int32) for q in self.polys[i]]
            cv2.fillPoly(mask, pts, 1)
            inside = g[mask > 0]
            if inside.size < 10:
                raise RuntimeError(f'{self.names[i]}: 航空写真の範囲に形が入っていない')
            lo, hi = np.percentile(inside, [2, 98])
            gn = np.clip((g - lo) / max(hi - lo, 1), 0, 1)
            self.d[i] = (img, gn, Mpx, z)
            self.zooms.add(z)
        return self.d[i]


def poly_px(M, polys, scale=16):
    return [np.round((h3(M) @ np.c_[q, np.ones(len(q))].T)[:2].T * scale).astype(np.int32) for q in polys]


def main_panel(n, cs, inv, tf, draw, polys, aer):
    lo, hi = TONE[inv]
    canvas = np.full((H, MW), 255.0 if inv else 0.0, np.float32)
    for c in cs:
        parts = draw[c]
        texs, masks = [], []
        for i, mir, a, w in parts:
            if mir:
                raise ValueError('鏡像の割り当てが混ざっている（--no-mirror の割り当てを使うこと）')
            _, gn, Mpx, _ = aer.get(i)
            M = pose_affine(polys[i], ANG[a], tf[c])
            T = (h3(M) @ np.linalg.inv(h3(Mpx)))[:2]
            shrink = abs(np.linalg.det(T[:, :2])) < 1
            texs.append(cv2.warpAffine(gn, T, (MW, H), flags=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REPLICATE))
            m = np.zeros((H, MW), np.uint8)
            cv2.fillPoly(m, poly_px(M, polys[i]), 255, lineType=cv2.LINE_AA, shift=4)
            masks.append(m)
        if len(parts) == 1:
            alpha = masks[0] / 255.0; tex = texs[0]
        else:
            acc = sum(w * sdf(m > 127) for (_, _, _, w), m in zip(parts, masks))
            alpha = cv2.GaussianBlur((acc < 0).astype(np.float32), (3, 3), 0)
            tex = sum(w * t for (_, _, _, w), t in zip(parts, texs))
        toned = lo + tex * (hi - lo)
        canvas = canvas * (1 - alpha) + toned * alpha
    return cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def split_name(nm, kind='市区町村'):
    if kind != '市区町村':
        return kind, nm
    for p in ('東京都', '北海道', '大阪府', '京都府'):
        if nm.startswith(p):
            return p, nm[len(p):]
    k = nm.find('県')
    return (nm[:k + 1], nm[k + 1:]) if 0 < k <= 3 else ('', nm)


MAX_CELLS = 3
MX = 8                          # 右の帯の左右の余白
THUMB_W, THUMB_H = SW - 2 * MX, (SW - 2 * MX) * 3 // 4   # 元の映像のサムネイル（4:3）
CELL_Y0, CELL_Y1, CELL_GAP = MX + THUMB_H + 8, H - 20, 8


class Side:
    """右の帯。映っている市区町村（最大 MAX_CELLS）ごとに「名前・航空写真（北が上）・位置図」の区画を縦に並べる。
    並びの組ごとに固定部分をキャッシュする。"""
    def __init__(self, names, polys, geoms, aer, kind):
        self.names, self.polys, self.geoms, self.aer, self.kind = names, polys, geoms, aer, kind
        self.base_map, self.xy = japan_map(geoms, 260, 260)
        self.f = {k: font(k) for k in (11, 13, 14, 16, 18, 22, 30)}
        self.cache = {}; self.map_cache = {}

    def mini_map(self, m, size):
        key = (m, size)
        if key not in self.map_cache:
            mp = self.base_map.copy(); dm = ImageDraw.Draw(mp); g = self.geoms[m]
            for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
                q = np.array(p.exterior.coords); q = self.xy(q[:, 0], q[:, 1])
                dm.polygon([tuple(t) for t in q], fill=(235, 70, 70))
            cc = self.xy(np.array([g.centroid.x]), np.array([g.centroid.y]))[0]
            dm.ellipse([cc[0] - 9, cc[1] - 9, cc[0] + 9, cc[1] + 9], outline=(255, 90, 90), width=3)
            self.map_cache[key] = mp.resize((size, size), Image.LANCZOS)
        return self.map_cache[key]

    def photo(self, img, m, box):
        color, _, Mpx, _ = self.aer.get(m)
        M0 = fit_affine(self.polys[m], box)
        T = (h3(M0) @ np.linalg.inv(h3(Mpx)))[:2]
        tex = cv2.warpAffine(color, T, (SW, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        mask = np.zeros((H, SW), np.uint8)
        pts = poly_px(M0, self.polys[m])
        cv2.fillPoly(mask, pts, 255, lineType=cv2.LINE_AA, shift=4)
        a = (mask / 255.0)[..., None]
        img[:] = (tex * a + img * (1 - a)).astype(np.uint8)
        cv2.polylines(img, pts, True, (255, 255, 255), 2, lineType=cv2.LINE_AA, shift=4)

    def static(self, ms, extra=0):
        key = (ms, extra)
        if key in self.cache:
            return self.cache[key]
        img = np.full((H, SW, 3), 22, np.uint8)
        k = len(ms)
        cells = []
        if k:
            h = (CELL_Y1 - CELL_Y0 - CELL_GAP * (k - 1)) / k
            for j, m in enumerate(ms):
                y = int(CELL_Y0 + j * (h + CELL_GAP)); hh = 54 if k == 1 else 30
                body = int(h) - hh
                if k == 1:
                    ms_ = 140
                    self.photo(img, m, (MX, y + hh, THUMB_W, body - ms_ - 6))
                    cells.append((m, y, hh, (SW - MX - ms_, y + int(h) - ms_), ms_))
                else:
                    ms_ = min(body, 110)
                    self.photo(img, m, (MX, y + hh, THUMB_W - ms_ - 6, body))
                    cells.append((m, y, hh, (SW - MX - ms_, y + hh + (body - ms_) // 2), ms_))
                if j:
                    cv2.line(img, (MX, y - CELL_GAP // 2), (SW - MX, y - CELL_GAP // 2), (60, 60, 60), 1)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for m, y, hh, mpos, msz in cells:
            pil.paste(self.mini_map(m, msz), mpos)
            pref, rest = split_name(self.names[m], self.kind[m])
            if k == 1:
                d.text((MX, y), pref, font=self.f[18], fill=(200, 200, 200))
                d.text((MX, y + 20), rest, font=self.f[30] if len(rest) <= 9 else self.f[22], fill=(255, 255, 255))
            else:
                big = self.f[22] if len(rest) <= 7 else self.f[16]
                pw = d.textlength(pref, font=self.f[13]) if pref else 0
                d.text((MX, y + 8), pref, font=self.f[13], fill=(200, 200, 200))
                d.text((MX + pw + (4 if pref else 0), y + (2 if big is self.f[22] else 6)), rest, font=big, fill=(255, 255, 255))
        if extra:
            d.text((SW - 80, H - 17), f'ほか {extra} か所', font=self.f[11], fill=(200, 200, 200))
        d.text((MX, H - 17), '画像：国土地理院（地理院タイル）', font=self.f[11], fill=(120, 120, 120))
        self.cache[key] = np.array(pil)
        return self.cache[key]


MIN_SHOW = 4                    # 右の帯に出す最短の連続フレーム数（最大の形は短くても出す）
SIDE_FADE = 4                   # カット以外で組が変わるときの切り替えフレーム数（消す 2 + 出す 2）。カットでは即座に切り替える


def side_lists(z, draw, by_frame, tf, lo, hi):
    """各フレームで右の帯に並べる市区町村の組と、前の組からの切り替え比率。
    その瞬間に主画面にある形だけを出す（前の表示は残さない）。一瞬だけ出入りする形でレイアウトが揺れないよう、
    MIN_SHOW フレーム以上続けて映っている形に限る。ただし最大の形は必ず出す。
    並び順は、新しく加わったものを主画面で左にある順に後ろへ足す（既にあるものの順は保つ）。"""
    present = {n: {} for n in range(1, N_FRAMES + 1)}       # n → {市区町村: (面積, x)}
    for n, cs in by_frame.items():
        for c in cs:
            m = draw[c][-1][0]; a = int(z['area'][c])
            if m not in present[n] or a > present[n][m][0]:
                present[n][m] = (a, float(tf[c][1]))
    elig = {n: set() for n in present}
    runs = {}
    for n in range(1, N_FRAMES + 2):
        cur = set(present.get(n, {}))
        for m in list(runs):
            if m not in cur:
                s = runs.pop(m)
                if n - s >= MIN_SHOW:
                    for t in range(s, n):
                        elig[t].add(m)
        for m in cur:
            runs.setdefault(m, n)
    out = {}; lst = []; prev_key = cur_key = None; since = FADE
    for n in range(1, hi + 1):
        P = present.get(n, {})
        E = set(elig[n])
        if P:
            E.add(max(P, key=lambda m: P[m][0]))
        keep = [m for m in lst if m in E]
        new = sorted(E - set(keep), key=lambda m: P[m][1])
        full = keep + new
        if len(full) > MAX_CELLS:
            top = set(sorted(full, key=lambda m: -P[m][0])[:MAX_CELLS])
            shown = [m for m in full if m in top]
        else:
            shown = full
        lst = full
        key = (tuple(shown), len(full) - len(shown))
        if key != cur_key:
            instant = cur_key is None or bool(z['cut'][n - 1])
            prev_key, cur_key, since = cur_key if cur_key is not None else key, key, (SIDE_FADE if instant else 0)
        since += 1
        if n >= lo:
            out[n] = (prev_key, cur_key, min(1.0, since / SIDE_FADE))
    return out


def credits(zooms, f):
    lines = ['使用したデータ', '',
             '行政区域：国土数値情報（行政区域データ）（国土交通省）を',
             '　smartnews-smri/japan-topography が簡素化したものを加工して作成', '']
    if any(z >= 14 for z in zooms):
        lines += [PHOTO_CREDIT]
    if any(z <= 13 for z in zooms):
        lines += ['衛星画像：国土地理院（地理院タイル）', '　Landsat8画像（GSI,TSIC,GEO Grid/AIST）',
                  '　Landsat8画像（courtesy of the U.S. Geological Survey）']
    img = Image.new('RGB', (W, H), (0, 0, 0)); d = ImageDraw.Draw(img)
    y = H // 2 - len(lines) * 15
    for ln in lines:
        d.text((160, y), ln, font=f, fill=(230, 230, 230)); y += 30
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else N_FRAMES
    z = np.load(OBJ)
    pl = json.load(open(PLAN))
    names, polys, _ = load_munis()
    _, geoms, kind, _ = load_geoms()
    tf, draw, by_frame, label_alpha = build_draw(z, pl)
    aer = Aerials(names, polys, geoms)
    side = Side(names, polys, geoms, aer, kind)
    labels = side_lists(z, draw, by_frame, tf, lo, hi)

    out = os.path.join(BASE, 'out', f'video_c_{lo:04d}_{hi:04d}.mp4')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    if not vw.isOpened():
        raise RuntimeError('VideoWriter を開けない: ' + out)
    t0 = time.time()
    for n in range(lo, hi + 1):
        inv = bool(z['inv'][n - 1])
        cs = sorted(by_frame.get(n, []), key=lambda c: -z['area'][c])
        mainp = main_panel(n, cs, inv, tf, draw, polys, aer)
        prev, cur, al = labels[n]
        st = side.static(*cur)
        if al < 1:                      # 文字が重ならないよう、前の組を消してから次の組を出す
            empty = side.static((), 0)
            src, k = (side.static(*prev), 1 - 2 * al) if al < 0.5 else (st, 2 * al - 1)
            st = (empty * (1 - k) + src * k).astype(np.uint8)
        sp = Image.fromarray(st.copy())
        raw = Image.open(os.path.join(FRAME_DIR, f'frame-{n:04d}.jpg')).convert('RGB').resize((THUMB_W, THUMB_H), Image.LANCZOS)
        sp.paste(raw, (MX, MX))
        frame_img = np.hstack([mainp, cv2.cvtColor(np.array(sp), cv2.COLOR_RGB2BGR)])
        vw.write(frame_img)
        if (n - lo) % 500 == 0:
            print(f'  frame {n}  {time.time() - t0:.0f}s', flush=True)
    if hi == N_FRAMES:
        cr = credits(aer.zooms, font(20))
        for _ in range(CREDIT_SEC * FPS):
            vw.write(cr)
    vw.release()
    print('saved', out, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
