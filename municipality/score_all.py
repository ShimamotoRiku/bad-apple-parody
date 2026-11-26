"""全フレーム × 全市区町村（48 角度 × 左右反転）の IoU を正規化空間でまとめて計算する。

1. bank: 各市区町村を重心・面積でそろえて G×G にラスタライズしたビットマスク（data/bank.npz）
2. scores: 各フレームの図形を同じく正規化し、ビット演算で全候補との IoU を計算（data/scores.npz）
   保存するのは市区町村×反転ごとの最大 IoU と、そのときの角度番号。

使い方: python score_all.py [開始フレーム] [終了フレーム]   （既定は全 6572 フレーム）
"""
import os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from common import BASE, G, ANG, WORDS, N_FRAMES, load_munis, canon_points, raster_canon, pack, load_frame, canon_target

BANK = os.path.join(BASE, 'data', 'bank.npz')


def build_bank():
    names, polys, _ = load_munis()
    N = len(names)
    bits = np.zeros((N, 2, len(ANG), WORDS), np.uint64)
    t0 = time.time(); clipped = 0
    for i in range(N):
        for mir in (0, 1):
            for a, ang in enumerate(ANG):
                pts = canon_points(polys[i], ang, mir)
                allp = np.vstack(pts)
                if allp.min() < 0 or allp.max() > G:
                    clipped += 1          # グリッドからはみ出す姿勢は照合しない（切れた形で高得点になるのを防ぐ）
                    continue
                bits[i, mir, a] = pack(raster_canon(pts))
    cnt = np.bitwise_count(bits).sum(-1).astype(np.int32)
    np.savez(BANK, bits=bits, cnt=cnt)
    print(f'bank: {N} 市区町村 x 2 x {len(ANG)} 角度, はみ出しで除外 {clipped}, {time.time() - t0:.0f}s', flush=True)
    return bits, cnt


def load_bank():
    if not os.path.exists(BANK):
        return build_bank()
    z = np.load(BANK)
    return z['bits'], z['cnt']


def score_frame(n, bits, cnt):
    fig, inv = load_frame(n)
    T, tf = canon_target(fig)
    N = bits.shape[0]
    if T is None:
        return np.zeros((N, 2), np.float16), np.zeros((N, 2), np.uint8), inv, np.full(3, np.nan)
    t = pack(T)
    inter = np.bitwise_count(bits & t).sum(-1)              # (N, 2, A)
    iou = inter / np.maximum(cnt + int(T.sum()) - inter, 1)   # 散らばった島嶼部はグリッド外で面積0 → IoU 0
    am = iou.argmax(-1)
    return np.take_along_axis(iou, am[..., None], -1)[..., 0].astype(np.float16), am.astype(np.uint8), inv, np.array(tf)


def main():
    lo = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    hi = int(sys.argv[2]) if len(sys.argv) > 2 else N_FRAMES
    bits, cnt = load_bank()
    frames = list(range(lo, hi + 1))
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda n: score_frame(n, bits, cnt), frames))
    print(f'scores: {len(frames)} フレーム, {time.time() - t0:.0f}s', flush=True)
    out = os.path.join(BASE, 'data', f'scores_{lo:04d}_{hi:04d}.npz')
    np.savez(out, frames=np.array(frames), iou=np.stack([r[0] for r in res]), ang=np.stack([r[1] for r in res]),
             inv=np.array([r[2] for r in res]), tf=np.stack([r[3] for r in res]))
    print('saved', out)


if __name__ == '__main__':
    main()
