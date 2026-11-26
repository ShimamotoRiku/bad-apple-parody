"""見せ方の試作（静止画）。市区町村が分かりやすくなるよう、航空写真・位置・回転角を添える。

案A: シルエットの中をその市区町村の航空写真で塗る（回転のみ。鏡像は使わない）
案B: シルエットは白黒のまま、横のカードに航空写真（北が上）を添える
案C: 案A の写真をモノクロにし、図形の白黒に寄せる（Bad Apple!! の白黒の見た目を保つ）
どちらも右の帯に「元の映像」「名前」「実際の形（北が上）」「回転角」「日本地図上の位置」を出す。

使い方: python design_proto.py 1259 2538 4135 4455   → out/design_A_XXXX.png, out/design_B_XXXX.png
"""
import os, sys, json, pickle
import numpy as np
import cv2
from PIL import Image, ImageDraw
from common import BASE, G, ANG, FRAME_DIR, load_munis, poly_area_centroid, A0
from render import font, japan_map
from aerial import mosaic, local_to_px_affine, PHOTO_CREDIT, LANDSAT_CREDIT

W, H = 1280, 720
MW = 960                       # 主画面の幅（960x720）
FW = 240
SC = MW / FW
PLAN = os.path.join(BASE, 'data', 'oplan_3.0_nomir.json')


def h3(M):
    return np.vstack([M, [0, 0, 1]])


def pose_affine(polys, ang, tf):
    """局所座標 p → 主画面の画素。canon_points（反転なし）と objects.py の描画と同じ変換"""
    A, C = poly_area_centroid(polys)
    s0 = np.sqrt(A0 / A)
    s, cx, cy = tf
    c, sn = np.cos(ang), np.sin(ang)
    R = np.array([[c, -sn], [sn, c]]) * (s0 * SC / s)
    t = -R @ C + np.array([cx, cy]) * SC
    return np.c_[R, t]


def fit_affine(polys, box):
    """北が上のまま box (x, y, w, h) に収める変換"""
    allp = np.vstack(polys); mn, mx = allp.min(0), allp.max(0)
    x, y, w, h = box
    k = min(w / (mx - mn)[0], h / (mx - mn)[1]) * 0.92
    ctr = (mn + mx) / 2
    return np.c_[np.eye(2) * k, -k * ctr + [x + w / 2, y + h / 2]]


def tone_map(tex, tone):
    """案C: 写真をモノクロにし、図形の色（white=明るい側 / black=暗い側）に寄せる"""
    g = cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo_, hi_ = np.percentile(g, [2, 98]); g = np.clip((g - lo_) / max(hi_ - lo_, 1), 0, 1)
    g = 150 + g * 105 if tone == 'white' else g * 110
    return cv2.cvtColor(g.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def textured(canvas_bgr, polys, M_local, geom_cache_entry, outline, tone=None):
    """M_local: 局所座標 → canvas 画素。航空写真を貼って輪郭を引く"""
    img, Mpx, z = geom_cache_entry
    M = (h3(M_local) @ np.linalg.inv(h3(Mpx)))[:2]           # モザイク画素 → canvas 画素
    hh, ww = canvas_bgr.shape[:2]
    tex = cv2.warpAffine(img, M, (ww, hh), flags=cv2.INTER_AREA if abs(np.linalg.det(M[:, :2])) < 1 else cv2.INTER_LINEAR)
    if tone is not None:
        tex = tone_map(tex, tone)
    mask = np.zeros((hh, ww), np.uint8)
    pts = [np.round((h3(M_local) @ np.c_[q, np.ones(len(q))].T)[:2].T * 16).astype(np.int32) for q in polys]
    cv2.fillPoly(mask, pts, 255, lineType=cv2.LINE_AA, shift=4)
    a = (mask / 255.0)[..., None]
    canvas_bgr[:] = (tex * a + canvas_bgr * (1 - a)).astype(np.uint8)
    if outline is not None:
        cv2.polylines(canvas_bgr, pts, True, outline, 2, lineType=cv2.LINE_AA, shift=4)
    return z


def main():
    frames = [int(a) for a in sys.argv[1:]] or [1259, 2538, 4135, 4455]
    z = np.load(os.path.join(BASE, 'data', 'objects.npz'))
    pl = json.load(open(PLAN))
    names, polys, _ = load_munis()
    geoms = pickle.load(open(os.path.join(BASE, 'data', 'muni.pkl'), 'rb'))['geoms']
    base_map, xy = japan_map(geoms, 260, 260)
    f_big, f_mid, f_small, f_tiny = font(30), font(18), font(14), font(11)
    aer = {}
    for n in frames:
        cs = sorted(np.flatnonzero(z['frame'] == n), key=lambda c: -z['area'][c])
        if not cs:
            raise ValueError(f'frame {n} に物体がない')
        inv = bool(z['inv'][n - 1])
        for c in cs:
            i = pl[c][0]
            if i not in aer:
                img, to_px, zz = mosaic(geoms[i])
                aer[i] = (img, local_to_px_affine(geoms[i], to_px), zz)
                print(f'  {names[i]}: zoom {zz}, モザイク {img.shape[1]}x{img.shape[0]}', flush=True)
        for variant in 'ABC':
            bgv = 255 if inv else 0
            main_ = np.full((H, MW, 3), bgv, np.uint8)
            zooms = set()
            for c in cs:
                i, mir, a, v = pl[c]
                M = pose_affine(polys[i], ANG[a], z['tf'][c])
                if variant == 'A':
                    zooms.add(textured(main_, polys[i], M, aer[i], (235, 235, 235) if not inv else (40, 40, 40)))
                elif variant == 'C':
                    zooms.add(textured(main_, polys[i], M, aer[i], None, tone='black' if inv else 'white'))
                else:
                    pts = [np.round((h3(M) @ np.c_[q, np.ones(len(q))].T)[:2].T * 16).astype(np.int32) for q in polys[i]]
                    cv2.fillPoly(main_, pts, (0, 0, 0) if inv else (255, 255, 255), lineType=cv2.LINE_AA, shift=4)
            # 右の帯
            side = np.full((H, W - MW, 3), 22, np.uint8)
            raw = cv2.resize(cv2.imread(os.path.join(FRAME_DIR, f'frame-{n:04d}.jpg')), (288, 216), interpolation=cv2.INTER_AREA)
            side[30:246, 16:304] = raw
            c0 = cs[0]; i0, _, a0, _ = pl[c0]
            card = (16, 330, 288, 200)
            M0 = fit_affine(polys[i0], card)
            zooms.add(textured(side, polys[i0], M0, aer[i0], (255, 255, 255)))
            canvas = Image.fromarray(cv2.cvtColor(np.hstack([main_, side]), cv2.COLOR_BGR2RGB))
            # 位置図
            mp = base_map.copy(); dm = ImageDraw.Draw(mp)
            for rank, c in enumerate(cs):
                g = geoms[pl[c][0]]
                for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
                    q = np.array(p.exterior.coords); q = xy(q[:, 0], q[:, 1])
                    dm.polygon([tuple(t) for t in q], fill=(235, 70, 70) if rank == 0 else (240, 170, 60))
                cc = xy(np.array([g.centroid.x]), np.array([g.centroid.y]))[0]
                dm.ellipse([cc[0] - 7, cc[1] - 7, cc[0] + 7, cc[1] + 7], outline=(255, 90, 90) if rank == 0 else (240, 170, 60), width=2)
            canvas.paste(mp.resize((150, 150)), (W - 166, 548))
            d = ImageDraw.Draw(canvas)
            d.text((MW + 16, 10), '元の映像', font=f_small, fill=(160, 160, 160))
            nm = names[i0]
            pref = nm[:nm.index('県') + 1] if '県' in nm[:4] else (nm[:3] if nm[:3] in ('東京都', '北海道', '大阪府', '京都府') else '')
            d.text((MW + 16, 256), pref, font=f_mid, fill=(200, 200, 200))
            d.text((MW + 16, 280), nm[len(pref):], font=f_big if len(nm) - len(pref) <= 9 else f_mid, fill=(255, 255, 255))
            d.text((MW + 16, 534), f'実際の形（北が上）　時計回りに {round(np.degrees(ANG[a0]))}° 回して配置', font=f_tiny, fill=(170, 170, 170))
            if len(cs) > 1:
                y = 560
                d.text((MW + 16, y), 'ほかに映っている形', font=f_tiny, fill=(170, 170, 170))
                for c in cs[1:4]:
                    y += 18; d.text((MW + 16, y), '・' + names[pl[c][0]], font=f_tiny, fill=(240, 170, 60))
            d.text((MW + 16, H - 18), '画像：国土地理院（地理院タイル）', font=f_tiny, fill=(120, 120, 120))   # 詳しい出典は動画末尾と概要欄
            out = os.path.join(BASE, 'out', f'design_{variant}_{n:04d}.png')
            canvas.save(out)
            print('saved', out)


if __name__ == '__main__':
    main()
