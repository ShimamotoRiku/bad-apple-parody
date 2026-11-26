# 高速照合の共通部品: 市区町村形状の正規化・ラスタライズ、フレームの読み込みと正規化
import os, pickle
import numpy as np
import cv2
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
FRAME_DIR = os.path.join(BASE, 'frames', 'bad-apple', 'src', 'public', 'images', 'frames')
N_FRAMES = 6572

G = 64                 # 正規化グリッド（G×G）
A0 = 0.15 * G * G      # 正規化後の図形面積（px）
ANG = np.radians(np.arange(0, 360, 7.5))   # 48 角度
WORDS = G * G // 64


def load_geoms():
    """市区町村（muni.pkl）に、都道府県・地方（areas.pkl、prep_areas.py で作成）を後ろに足した一覧。
    戻り値: names, geoms, kind（'市区町村' | '都道府県' | '地方'）, adj（市区町村のみ。都道府県・地方は空集合）"""
    D = pickle.load(open(os.path.join(BASE, 'data', 'muni.pkl'), 'rb'))
    names, geoms, adj = list(D['names']), list(D['geoms']), list(D['adj'])
    kind = ['市区町村'] * len(names)
    ap = os.path.join(BASE, 'data', 'areas.pkl')
    if not os.path.exists(ap):
        raise FileNotFoundError('data/areas.pkl がない。先に python prep_areas.py を実行する')
    A = pickle.load(open(ap, 'rb'))
    names += A['names']; geoms += A['geoms']; kind += A['kind']; adj += [set() for _ in A['names']]
    return names, geoms, kind, adj


def load_munis():
    names, geoms, kind, adj = load_geoms()
    polys = []
    for g, kd in zip(geoms, kind):
        ps = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
        amax = max(p.area for p in ps)
        k = np.cos(np.radians(g.centroid.y))
        # 小さい離島は除外（市区町村は最大パーツの5%未満、都道府県・地方は20%未満。鹿児島県の奄美などで形が散らばるため）。
        # 経度を cos(緯度) 補正し、北を上（y 反転）にする
        th = 0.05 if kd == '市区町村' else 0.2
        polys.append([np.c_[np.array(p.exterior.coords)[:, 0] * k, -np.array(p.exterior.coords)[:, 1]]
                      for p in ps if p.area > th * amax])
    return names, polys, adj


def poly_area_centroid(polys):
    A = 0.0; C = np.zeros(2)
    for q in polys:
        x, y = q[:, 0], q[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        a = cr.sum() / 2
        if abs(a) < 1e-18:
            continue
        c = np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6 * a)
        A += abs(a); C += abs(a) * c
    if A == 0:
        raise ValueError('面積ゼロの形状')
    return A, C / A


def canon_points(polys, ang, mir):
    """市区町村を回転・反転し、重心をグリッド中心、面積を A0 にそろえた座標（G×G 上）を返す"""
    A, C = poly_area_centroid(polys)
    s = np.sqrt(A0 / A)
    c, sn = np.cos(ang), np.sin(ang)
    out = []
    for q in polys:
        p = (q - C) * s
        if mir:
            p = p * [-1, 1]
        out.append(np.c_[c * p[:, 0] - sn * p[:, 1], sn * p[:, 0] + c * p[:, 1]] + G / 2)
    return out


def raster_canon(pts):
    img = np.zeros((G, G), np.uint8)
    cv2.fillPoly(img, [np.round((p - 0.5) * 16).astype(np.int32) for p in pts], 1, lineType=cv2.LINE_8, shift=4)
    return img.astype(bool)


def pack(mask):
    return np.packbits(mask.ravel()).view(np.uint64)


def load_frame(n, size=(240, 180)):
    """フレーム n（1始まり）を読み、図形側を True にした二値画像と、反転したかどうかを返す。
    図形側の判定: 画面の縁で多い色を背景とみなす。"""
    path = os.path.join(FRAME_DIR, f'frame-{n:04d}.jpg')
    im = np.array(Image.open(path).convert('L').resize(size, Image.BILINEAR)) > 127
    border = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]])
    inverted = border.mean() > 0.5
    fig = ~im if inverted else im
    return fig, inverted


def canon_target(fig):
    """二値画像の図形を重心・面積で正規化して G×G に写す。図形が空なら None。
    戻り値: (正規化マスク, 変換パラメータ (s, cx, cy)) — 元画像の連続座標 = (canon - G/2)/s + (cx, cy)"""
    m = cv2.moments(fig.astype(np.uint8), binaryImage=True)
    if m['m00'] < 4:
        return None, None
    cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
    s = np.sqrt(A0 / m['m00'])
    # 画素 j の中心は連続座標 j+0.5。連続座標で X = s*(x - (cx+0.5)) + G/2 となるよう、画素添字どうしの写像を作る
    M = np.float32([[s, 0, G / 2 - 0.5 - s * cx], [0, s, G / 2 - 0.5 - s * cy]])
    w = cv2.warpAffine(fig.astype(np.float32), M, (G, G), flags=cv2.INTER_LINEAR)
    return w > 0.5, (s, cx + 0.5, cy + 0.5)


def canon_region(fig, valid_full):
    """canon_target と同じ正規化に加え、画面内（valid_full=True の範囲）を正規化空間に写したマスクも返す。
    画面外は「どちらでもない」として IoU から除外するために使う。"""
    m = cv2.moments(fig.astype(np.uint8), binaryImage=True)
    if m['m00'] < 4:
        return None, None, None
    cx, cy = m['m10'] / m['m00'], m['m01'] / m['m00']
    s = np.sqrt(A0 / m['m00'])
    M = np.float32([[s, 0, G / 2 - 0.5 - s * cx], [0, s, G / 2 - 0.5 - s * cy]])
    w = cv2.warpAffine(fig.astype(np.float32), M, (G, G), flags=cv2.INTER_LINEAR)
    v = cv2.warpAffine(valid_full.astype(np.float32), M, (G, G), flags=cv2.INTER_LINEAR, borderValue=0)
    return (w > 0.5) & (v > 0.5), v > 0.5, (s, cx + 0.5, cy + 0.5)
