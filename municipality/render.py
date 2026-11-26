"""割り当て（data/plan_*.json）からプレビュー動画を作る（音声なし）。

左: 原画（二値化前のフレーム）、右: 市区町村の形、下: 名前と IoU、右下: 位置を示す小地図。
使い方: python render.py data/plan_1.0.json [開始フレーム] [終了フレーム]
出力: out/preview_<plan名>_<開始>_<終了>.mp4
"""
import os, sys, json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from common import BASE, G, ANG, FRAME_DIR, load_munis, canon_points
import pickle

FW, FH = 240, 180          # score_all で使ったフレーム座標系
OW, OH = 640, 480          # 出力の片側サイズ
BAR = 56
FONT_PATHS = ['C:/Windows/Fonts/meiryob.ttc', 'C:/Windows/Fonts/YuGothB.ttc',
              '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc']


def font(size):
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    raise FileNotFoundError('日本語フォントが見つからない: ' + ', '.join(FONT_PATHS))


def japan_map(geoms, w=200, h=200):
    """小地図の下地と、経緯度→画素の変換を返す"""
    lon0, lon1, lat0, lat1 = 127.5, 146.0, 26.0, 45.8
    k = np.cos(np.radians(36))
    sx = w / ((lon1 - lon0) * k); sy = h / (lat1 - lat0); sc = min(sx, sy)

    def xy(lon, lat):
        return np.c_[(lon - lon0) * k * sc, (lat1 - lat) * sc]
    img = Image.new('RGB', (w, h), (30, 30, 30)); dr = ImageDraw.Draw(img)
    for g in geoms:
        for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
            a = np.array(p.exterior.coords)
            q = xy(a[:, 0], a[:, 1])
            if len(q) >= 3:
                dr.polygon([tuple(v) for v in q], fill=(110, 110, 110))
    return img, xy


def main():
    plan_path = sys.argv[1]
    plan = json.load(open(plan_path, encoding='utf-8'))
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else plan[0]['start']
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else plan[-1]['end']
    names, polys, _ = load_munis()
    geoms = pickle.load(open(os.path.join(BASE, 'data', 'muni.pkl'), 'rb'))['geoms']
    sc = np.load(os.path.join(BASE, 'data', 'scores_0001_6572.npz'))
    tf_of = {int(n): tf for n, tf in zip(sc['frames'], sc['tf'])}
    inv_of = {int(n): bool(v) for n, v in zip(sc['frames'], sc['inv'])}
    base_map, xy = japan_map(geoms)
    f_name, f_small = font(22), font(15)

    os.makedirs(os.path.join(BASE, 'out'), exist_ok=True)
    tag = os.path.splitext(os.path.basename(plan_path))[0]
    out = os.path.join(BASE, 'out', f'preview_{tag}_{lo:04d}_{hi:04d}.mp4')
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*'mp4v'), 30, (OW * 2, OH + BAR))
    if not vw.isOpened():
        raise RuntimeError('VideoWriter を開けない: ' + out)

    pts_cache = {}
    for seg in plan:
        if seg['end'] < lo or seg['start'] > hi:
            continue
        i, mir = seg['muni'], seg['mir']
        g = geoms[i]
        mp = base_map.copy(); dm = ImageDraw.Draw(mp)
        for p in (g.geoms if g.geom_type == 'MultiPolygon' else [g]):
            a = np.array(p.exterior.coords); q = xy(a[:, 0], a[:, 1])
            dm.polygon([tuple(v) for v in q], fill=(230, 60, 60))
        c = xy(np.array([g.centroid.x]), np.array([g.centroid.y]))[0]
        dm.ellipse([c[0] - 5, c[1] - 5, c[0] + 5, c[1] + 5], outline=(255, 80, 80), width=2)
        for k, n in enumerate(range(seg['start'], seg['end'] + 1)):
            if n < lo or n > hi:
                continue
            raw = Image.open(os.path.join(FRAME_DIR, f'frame-{n:04d}.jpg')).convert('RGB').resize((OW, OH), Image.BILINEAR)
            inv = inv_of[n]
            fg, bg = ((0, 0, 0), (255, 255, 255)) if inv else ((255, 255, 255), (0, 0, 0))
            right = Image.new('RGB', (OW, OH), bg); dr = ImageDraw.Draw(right)
            tf = tf_of[n]
            if not np.isnan(tf[0]):
                key = (i, mir, seg['ang'][k])
                if key not in pts_cache:
                    pts_cache[key] = canon_points(polys[i], ANG[seg['ang'][k]], mir)
                s, cx, cy = tf
                for q in pts_cache[key]:
                    pp = ((q - G / 2) / s + [cx, cy]) * (OW / FW)
                    dr.polygon([tuple(v) for v in pp], fill=fg)
            canvas = Image.new('RGB', (OW * 2, OH + BAR), (20, 20, 20))
            canvas.paste(raw, (0, 0)); canvas.paste(right, (OW, 0))
            canvas.paste(mp.resize((150, 150)), (OW * 2 - 160, OH - 160))
            dc = ImageDraw.Draw(canvas)
            dc.text((12, OH + 12), seg['name'] + ('（反転）' if mir else ''), font=f_name, fill=(255, 255, 255))
            dc.text((OW * 2 - 250, OH + 18), f'frame {n}  IoU {seg["iou"][k]:.2f}', font=f_small, fill=(180, 180, 180))
            vw.write(cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR))
    vw.release()
    print('saved', out)


if __name__ == '__main__':
    main()
