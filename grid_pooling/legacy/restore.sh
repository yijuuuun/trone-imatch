#!/bin/bash
# Colab 런타임 복구 (Drive가 마운트된 뒤 실행)
set -e
C=/content/drive/MyDrive/bearinguav_cache
cd /content
[ -d bearinguav/.git ] || git clone -q https://github.com/liukejia121/bearinguav.git
cd bearinguav
mkdir -p diag && cp $C/diag/*.py diag/ && touch diag/__init__.py
if [ ! -d Bearing_UAV/satellite_view ]; then
  cp $C/Bearing_UAV.zip /content/ && unzip -q -o /content/Bearing_UAV.zip -d /content/bw && rm /content/Bearing_UAV.zip
  F=$(find /content/bw -name best_model.pth -path "*satellite_view*" | head -1); mkdir -p Bearing_UAV
  cp -r "$(dirname $(dirname $F))"/* Bearing_UAV/; rm -rf /content/bw
fi
if [ ! -d /content/data/Bearing_UAV_90K ]; then
  mkdir -p /content/data && cp $C/Bearing_UAV_90K.zip /content/data/ && unzip -q -o /content/data/Bearing_UAV_90K.zip -d /content/data && rm /content/data/Bearing_UAV_90K.zip
fi
D=$(find /content/data -maxdepth 3 -type d -name city_rsi | head -1); ln -sfn "$(dirname $D)" /content/bearinguav/Bearing_UAV_90K
find /content/bearinguav -name best_model.pth
ls /content/bearinguav/Bearing_UAV_90K | head
echo RESTORE_DONE
