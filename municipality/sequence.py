"""ちらつき対策: 同じ市区町村をどこまで保持できるかを動的計画法で求める。

段階1（市区町村の割り当て）: 状態 = (市区町村, 反転)。各フレームの得点は角度について最大の IoU。
  切り替えるたびに lam を引き、合計を最大化する経路を Viterbi で求める。
段階2（角度の追従）: 保持区間ごとに、その市区町村だけ 48 角度の IoU を計算し直し、
  角度が 7.5° 変わるたびに mu を引く経路を求める（回転が飛ばないようにする）。

使い方:
  python sequence.py sweep                  # lam を変えて保持と IoU のトレードオフを表示
  python sequence.py plan 1.0               # lam=1.0 の割り当てを data/plan_1.0.json に保存
"""
import os, sys, json
import numpy as np
from common import BASE, G, ANG, N_FRAMES, load_munis, canon_points, raster_canon, pack, load_frame, canon_target

SCORES = os.path.join(BASE, 'data', 'scores_0001_6572.npz')
FPS = 30


def load_scores():
    z = np.load(SCORES)
    iou = np.nan_to_num(z['iou'].astype(np.float32))      # (T, N, 2)
    return z, iou.reshape(iou.shape[0], -1)                 # 状態 k = 市区町村*2 + 反転


def viterbi(S, lam):
    T, K = S.shape
    V = S[0].copy()
    stay = np.zeros((T, K), bool)
    prev_best = np.zeros(T, np.int64)
    for t in range(1, T):
        j = int(V.argmax()); sw = V[j] - lam
        stay[t] = V >= sw
        prev_best[t] = j
        V = np.where(stay[t], V, sw) + S[t]
    path = np.zeros(T, np.int64); path[-1] = V.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = path[t] if stay[t, path[t]] else prev_best[t]
    return path


def segments(path):
    cut = np.flatnonzero(np.diff(path)) + 1
    starts = np.r_[0, cut]; ends = np.r_[cut, len(path)]
    return list(zip(starts, ends))


def sweep():
    z, S = load_scores()
    valid = ~np.isnan(z['tf'][:, 0])
    print(f'{"lam":>6} {"区間数":>6} {"平均長(秒)":>10} {"中央長(秒)":>10} {"平均IoU":>8}  (各フレーム最良の平均 {S.max(1)[valid].mean():.3f})')
    for lam in [0, 0.3, 1, 2, 4, 8]:
        p = viterbi(S, lam)
        seg = segments(p); L = np.array([e - s for s, e in seg])
        v = S[np.arange(len(p)), p][valid].mean()
        print(f'{lam:6.1f} {len(seg):6d} {L.mean() / FPS:10.2f} {np.median(L) / FPS:10.2f} {v:8.3f}')


def angle_track(polys, Ts, mir, mu=0.01):
    """1 区間の角度追従。Ts: 正規化済み目標マスクのリスト（None は空フレーム）"""
    pts = [canon_points(polys, a, mir) for a in ANG]
    clipped = np.array([np.vstack(q).min() < 0 or np.vstack(q).max() > G for q in pts])   # グリッドからはみ出す角度は使わない
    bank = np.stack([pack(raster_canon(q)) for q in pts])
    cnt = np.bitwise_count(bank).sum(-1)
    A = len(ANG)
    sc = np.zeros((len(Ts), A))
    for t, T in enumerate(Ts):
        if T is None:
            continue
        inter = np.bitwise_count(bank & pack(T)).sum(-1)
        sc[t] = inter / np.maximum(cnt + T.sum() - inter, 1)
    sc[:, clipped] = -1
    d = np.abs(np.arange(A)[:, None] - np.arange(A)[None, :]); d = np.minimum(d, A - d)
    cost = mu * d                                            # 前の角度 → 次の角度
    V = sc[0].copy(); bp = np.zeros((len(Ts), A), np.int64)
    for t in range(1, len(Ts)):
        M = V[:, None] - cost
        bp[t] = M.argmax(0); V = M.max(0) + sc[t]
    a = np.zeros(len(Ts), np.int64); a[-1] = V.argmax()
    for t in range(len(Ts) - 1, 0, -1):
        a[t - 1] = bp[t, a[t]]
    return a, sc[np.arange(len(Ts)), a]


def plan(lam, mu=0.01):
    z, S = load_scores()
    names, polys, _ = load_munis()
    p = viterbi(S, lam)
    out = []; ious = []
    for s, e in segments(p):
        i, mir = divmod(int(p[s]), 2)
        frames = [int(n) for n in z['frames'][s:e]]
        Ts = [canon_target(load_frame(n)[0])[0] for n in frames]
        a, v = angle_track(polys[i], Ts, mir, mu)
        out.append({'start': frames[0], 'end': frames[-1], 'muni': i, 'name': names[i], 'mir': mir,
                    'ang': a.tolist(), 'iou': [round(float(x), 3) for x in v]})
        ious += [x for x, T in zip(v, Ts) if T is not None]
    path = os.path.join(BASE, 'data', f'plan_{lam}.json')
    json.dump(out, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    L = np.array([o['end'] - o['start'] + 1 for o in out])
    print(f'lam={lam}: {len(out)} 区間, 平均 {L.mean() / FPS:.2f} 秒, 角度追従後の平均 IoU {np.mean(ious):.3f}')
    names_used = {}
    for o in out:
        names_used[o['name']] = names_used.get(o['name'], 0) + o['end'] - o['start'] + 1
    print('使われた市区町村数', len(names_used), ' 長く使われた順:',
          ', '.join(f'{k} {v / FPS:.1f}s' for k, v in sorted(names_used.items(), key=lambda kv: -kv[1])[:8]))
    print('saved', path)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    if sys.argv[1] == 'sweep':
        sweep()
    elif sys.argv[1] == 'plan':
        plan(float(sys.argv[2]))
    else:
        print(__doc__); sys.exit(1)
