# テスト用シルエット（りんご・人型・上半身）を targets/ に描く
import os
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'targets'); os.makedirs(OUT, exist_ok=True)
S = 10
def new():
    im = Image.new('L', (64 * S, 48 * S), 0); return im, ImageDraw.Draw(im)
def P(l): return [(x * S, y * S) for x, y in l]

im, d = new()
d.ellipse(P([(18, 12), (34, 42)]), fill=255); d.ellipse(P([(30, 12), (46, 42)]), fill=255)
d.rectangle(P([(31, 4), (33, 14)]), fill=255); d.ellipse(P([(34, 4), (42, 10)]), fill=255)
im.save(os.path.join(OUT, 'apple.png'))

im, d = new()
d.ellipse(P([(27, 3), (37, 13)]), fill=255)
d.polygon(P([(26, 14), (38, 14), (40, 30), (24, 30)]), fill=255)
d.polygon(P([(26, 14), (16, 6), (14, 8), (25, 18)]), fill=255)
d.polygon(P([(38, 14), (50, 22), (48, 24), (38, 19)]), fill=255)
d.polygon(P([(25, 30), (31, 30), (24, 46), (19, 46)]), fill=255)
d.polygon(P([(33, 30), (39, 30), (46, 44), (41, 46)]), fill=255)
im.save(os.path.join(OUT, 'person.png'))

im, d = new()
d.ellipse(P([(24, 6), (40, 24)]), fill=255)
d.polygon(P([(24, 14), (20, 34), (22, 40), (28, 26)]), fill=255)
d.polygon(P([(40, 14), (44, 36), (40, 40), (36, 24)]), fill=255)
d.polygon(P([(18, 48), (22, 34), (30, 28), (36, 28), (44, 34), (48, 48)]), fill=255)
d.polygon(P([(40, 16), (44, 18), (40, 20)]), fill=255)
im.save(os.path.join(OUT, 'bust.png'))
