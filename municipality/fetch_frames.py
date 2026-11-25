"""Bad Apple!! のフレームを取得して照合用の白黒画像にする。

フレームは irqd/bad-apple リポジトリ（src/public/images/frames/frame-XXXX.jpg、960x720、全6572枚）から
必要な番号だけ sparse checkout で取る。著作物なのでこのリポジトリにはコミットしない（.gitignore 済み）。

使い方:
  python fetch_frames.py 900 4500            # 指定フレームを取得
  python fetch_frames.py 900 --invert        # 人物が黒の場面は --invert で人物側を白にする
出力: frames/raw/frame-XXXX.jpg, real/fXXXX.png（match.py の入力）
"""
import argparse, os, subprocess
from PIL import Image, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(BASE, 'frames', 'bad-apple')

ap = argparse.ArgumentParser()
ap.add_argument('frames', type=int, nargs='+')
ap.add_argument('--invert', action='store_true', help='白黒を反転（照合対象＝白にする）')
a = ap.parse_args()

if not os.path.isdir(REPO):
    subprocess.run(['git', 'clone', '--depth', '1', '--filter=blob:none', '--no-checkout',
                    'https://github.com/irqd/bad-apple', REPO], check=True)
    subprocess.run(['git', 'sparse-checkout', 'init', '--no-cone'], cwd=REPO, check=True)
paths = [f'/src/public/images/frames/frame-{n:04d}.jpg' for n in a.frames]
subprocess.run(['git', 'sparse-checkout', 'add', *paths], cwd=REPO, check=True)
subprocess.run(['git', 'checkout'], cwd=REPO, check=True)

os.makedirs(os.path.join(BASE, 'real'), exist_ok=True)
for n in a.frames:
    im = Image.open(os.path.join(REPO, f'src/public/images/frames/frame-{n:04d}.jpg')).convert('L')
    if a.invert:
        im = ImageOps.invert(im)
    im.save(os.path.join(BASE, 'real', f'f{n:04d}.png'))
    print('saved', f'real/f{n:04d}.png')
