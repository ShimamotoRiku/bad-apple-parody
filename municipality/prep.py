# 市区町村ポリゴンを読み込み、平面座標化・隣接グラフを作って保存
import json, pickle, os, numpy as np
BASE = os.path.dirname(os.path.abspath(__file__))
from shapely.geometry import shape
from shapely.strtree import STRtree

d = json.load(open(os.path.join(BASE, 'data/japan-topography/data/municipality/geojson/s0010/N03-21_210101.json'), encoding='utf-8'))
names, geoms = [], []
for f in d['features']:
    p = f['properties']
    if p.get('N03_004') is None or '所属未定' in (p.get('N03_004') or ''):
        continue
    g = shape(f['geometry'])
    if g.is_empty: continue
    nm = (p['N03_001'] or '') + (p['N03_003'] or '') + (p['N03_004'] or '')
    names.append(nm); geoms.append(g)

# 同名の重複フィーチャーを統合
from shapely.ops import unary_union
merged = {}
for n, g in zip(names, geoms):
    merged.setdefault(n, []).append(g)
names = list(merged); geoms = [unary_union(v) if len(v) > 1 else v[0] for v in merged.values()]

def to_xy(g):
    # 局所的な等距円筒図法（中心緯度でcos補正）
    lat0 = g.centroid.y
    k = np.cos(np.radians(lat0))
    polys = list(g.geoms) if g.geom_type == 'MultiPolygon' else [g]
    out = []
    for pg in polys:
        a = np.array(pg.exterior.coords)
        out.append(np.c_[a[:, 0] * k, a[:, 1]])
    return out, k

shapes = []
for g in geoms:
    polys, k = to_xy(g)
    shapes.append(polys)

# 隣接グラフ（境界がほぼ接するもの）
tree = STRtree(geoms)
adj = [set() for _ in geoms]
for i, g in enumerate(geoms):
    gb = g.buffer(0.002)
    for j in tree.query(gb):
        j = int(j)
        if j != i and gb.intersects(geoms[j]):
            adj[i].add(j); adj[j].add(i)

print(len(names), 'municipalities; mean degree', np.mean([len(a) for a in adj]))
pickle.dump({'names': names, 'geoms': geoms, 'adj': adj}, open(os.path.join(BASE, 'data/muni.pkl'), 'wb'))
