"""動画に音声を載せる（音声は再生しない。長さだけ確認する）。

使い方: python add_audio.py [音声ファイル] [ずらす秒数]
  音声ファイル: 既定は ../resources/original.mp4（元の PV。音声トラックだけ使う）
  ずらす秒数: 正なら音声を遅らせる、負なら音声の頭を削る。
    既定は -2/30 秒。フレーム画像 frame-N.jpg は元の PV の (N+1) 番目（0 始まり）のコマと一致し、
    こちらの動画では N-1 番目に置いているので、映像が 2 コマ早い。その分だけ音声の頭を削る。
入力: out/video_c_h264.mp4 / 出力: out/video_c_audio.mp4
動画より音声が短ければ無音で埋め、長ければ動画の長さで切る。
"""
import os, sys, json, subprocess
from common import BASE

VIDEO = os.path.join(BASE, 'out', 'video_c_h264.mp4')
OUT = os.path.join(BASE, 'out', 'video_c_audio.mp4')
DEFAULT_AUDIO = os.path.join(BASE, '..', 'resources', 'original.mp4')
DEFAULT_OFFSET = -2 / 30


def duration(path):
    r = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', path],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'ffprobe 失敗: {path}\n{r.stderr}')
    return float(json.loads(r.stdout)['format']['duration'])


def main():
    audio = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO
    offset = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OFFSET
    if not os.path.exists(audio):
        raise FileNotFoundError(audio)
    vd, ad = duration(VIDEO), duration(audio)
    print(f'動画 {vd:.2f} 秒 / 音声 {ad:.2f} 秒 / ずらし {offset:+.2f} 秒')
    if offset >= 0:
        af = f'adelay={int(offset * 1000)}:all=1,apad'
    else:
        af = f'atrim=start={-offset},asetpts=PTS-STARTPTS,apad'
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', VIDEO, '-i', audio, '-map', '0:v', '-map', '1:a',
           '-c:v', 'copy', '-af', af, '-c:a', 'aac', '-b:a', '192k', '-t', f'{vd:.3f}', OUT]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('ffmpeg 失敗:\n' + r.stderr)
    print('saved', OUT, f'{duration(OUT):.2f} 秒')


if __name__ == '__main__':
    main()
