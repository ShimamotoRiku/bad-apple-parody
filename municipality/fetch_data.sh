#!/usr/bin/env bash
# 市区町村境界データ（国土数値情報 N03 2021年版を smartnews-smri/japan-topography が簡素化したもの）を取得
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
if [ ! -d data/japan-topography ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/smartnews-smri/japan-topography data/japan-topography
fi
cd data/japan-topography
MSYS_NO_PATHCONV=1 git sparse-checkout set --no-cone '/data/municipality/geojson/s0010/N03-21_210101.json'
echo "done: data/japan-topography/data/municipality/geojson/s0010/N03-21_210101.json"
