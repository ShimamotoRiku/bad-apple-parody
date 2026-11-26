"""市区町村の境界を合体させて、都道府県（47）と地方区分（8 地方。北海道は都道府県と同じなので除く）の形を作る。
出力: data/areas.pkl {'names': [...], 'geoms': [...], 'kind': ['都道府県' | '地方', ...]}
"""
import os, pickle
from shapely.ops import unary_union
from common import BASE

PREFS = ['北海道', '青森県', '岩手県', '宮城県', '秋田県', '山形県', '福島県', '茨城県', '栃木県', '群馬県', '埼玉県', '千葉県',
         '東京都', '神奈川県', '新潟県', '富山県', '石川県', '福井県', '山梨県', '長野県', '岐阜県', '静岡県', '愛知県', '三重県',
         '滋賀県', '京都府', '大阪府', '兵庫県', '奈良県', '和歌山県', '鳥取県', '島根県', '岡山県', '広島県', '山口県', '徳島県',
         '香川県', '愛媛県', '高知県', '福岡県', '佐賀県', '長崎県', '熊本県', '大分県', '宮崎県', '鹿児島県', '沖縄県']
REGIONS = {
    '東北地方': PREFS[1:7], '関東地方': PREFS[7:14], '中部地方': PREFS[14:23], '近畿地方': PREFS[23:30],
    '中国地方': PREFS[30:35], '四国地方': PREFS[35:39], '九州地方': PREFS[39:47],
}


def clean(g):
    # 簡素化した境界どうしの細い隙間・重なりを消す（約 100m）
    return g.buffer(0.001).buffer(-0.001)


def main():
    D = pickle.load(open(os.path.join(BASE, 'data', 'muni.pkl'), 'rb'))
    by_pref = {p: [] for p in PREFS}
    for n, g in zip(D['names'], D['geoms']):
        p = next((p for p in PREFS if n.startswith(p)), None)
        if p is None:
            raise ValueError(f'都道府県が分からない: {n}')
        by_pref[p].append(g)
    pg = {}
    for p in PREFS:
        if not by_pref[p]:
            raise ValueError(f'市区町村がない: {p}')
        pg[p] = clean(unary_union(by_pref[p]))
    names = list(PREFS); geoms = [pg[p] for p in PREFS]; kind = ['都道府県'] * len(PREFS)
    for r, ps in REGIONS.items():
        names.append(r); geoms.append(clean(unary_union([pg[p] for p in ps]))); kind.append('地方')
    pickle.dump({'names': names, 'geoms': geoms, 'kind': kind}, open(os.path.join(BASE, 'data', 'areas.pkl'), 'wb'))
    print(len(names), '個（都道府県', len(PREFS), '・地方', len(REGIONS), '）')


if __name__ == '__main__':
    main()
