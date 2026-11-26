"""地理院タイル（全国最新写真（シームレス））から市区町村の航空写真モザイクを作る。

- 出典表記が必要: 「国土地理院（地理院タイル）」。ズーム 13 以下は写真ではなくランドサット衛星画像のモザイクで、
  出典が別になる（LANDSAT_CREDIT）。
- 標準地図・淡色地図は、事前描画した動画への利用に測量法の申請が要る場合があるので使わない（地図は境界データで自前で描く）。
- 取得したタイルは data/tiles/ にキャッシュし、同じタイルは二度取らない。取得の間隔は FETCH_INTERVAL 秒あける。
"""
import os, time, math, urllib.request, urllib.error
import numpy as np
import cv2
from common import BASE

URL = 'https://cyberjapandata.gsi.go.jp/xyz/seamlessphoto/{z}/{x}/{y}.jpg'
CACHE = os.path.join(BASE, 'data', 'tiles')
FETCH_INTERVAL = 0.3
PHOTO_CREDIT = '航空写真：国土地理院（地理院タイル 全国最新写真）'
LANDSAT_CREDIT = '衛星画像：国土地理院（地理院タイル）Landsat8画像（GSI,TSIC,GEO Grid/AIST）, Landsat8画像（courtesy of the U.S. Geological Survey）'
UA = 'bad-apple-parody/0.1 (personal video experiment)'


def tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def fetch(z, x, y):
    path = os.path.join(CACHE, str(z), str(x), f'{y}.jpg')
    if os.path.exists(path):
        return cv2.imread(path)
    if os.path.exists(path + '.missing'):
        return None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(URL.format(z=z, x=x, y=y), headers={'User-Agent': UA})
    time.sleep(FETCH_INTERVAL)
    try:
        data = urllib.request.urlopen(req, timeout=30).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:                     # 海上などタイルがない場所
            open(path + '.missing', 'w').close()
            return None
        raise RuntimeError(f'タイル取得失敗 {z}/{x}/{y}: HTTP {e.code}') from e
    with open(path, 'wb') as f:
        f.write(data)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'タイルを画像として読めない {z}/{x}/{y}')
    return img


def choose_zoom(geom, target_px):
    lon0, lat0, lon1, lat1 = geom.bounds
    x0, y1 = tile_xy(lon0, lat0, 0); x1, y0 = tile_xy(lon1, lat1, 0)
    span = max(x1 - x0, y1 - y0) * 256          # ズーム 0 での画素数
    return int(np.clip(math.floor(math.log2(target_px / span)), 9, 17))


def mosaic(geom, target_px=900):
    """市区町村の外接矩形を覆うモザイク画像（BGR）と、(経度, 緯度) → 画素 の関数、使ったズームを返す"""
    z = choose_zoom(geom, target_px)
    lon0, lat0, lon1, lat1 = geom.bounds
    fx0, fy0 = tile_xy(lon0, lat1, z); fx1, fy1 = tile_xy(lon1, lat0, z)
    tx0, ty0, tx1, ty1 = int(fx0), int(fy0), int(fx1), int(fy1)
    img = np.full(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), 90, np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            t = fetch(z, tx, ty)
            if t is not None:
                img[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = t

    def to_px(lon, lat):
        x, y = tile_xy(lon, lat, z)
        return (x - tx0) * 256, (y - ty0) * 256
    return img, to_px, z


def local_to_px_affine(geom, to_px):
    """load_munis の局所座標 (lon*k, -lat) → モザイク画素 の近似アフィン（市区町村の範囲で 3 点合わせ）"""
    k = np.cos(np.radians(geom.centroid.y))
    lon0, lat0, lon1, lat1 = geom.bounds
    src = np.float32([[lon0 * k, -lat1], [lon1 * k, -lat1], [lon0 * k, -lat0]])
    dst = np.float32([to_px(lon0, lat1), to_px(lon1, lat1), to_px(lon0, lat0)])
    return cv2.getAffineTransform(src, dst)        # 2x3
