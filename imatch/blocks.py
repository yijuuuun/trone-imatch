"""파이프라인 ⑥: 후보 블록.

좌표 규칙
- 도시 지도 4096 px. 블록 (bx, by)는 지도의 [by*256, by*256+512) 행, [bx*256, bx*256+512) 열 (x가 열).
- 블록은 256 px(2단위)씩 어긋나 있어 이웃과 절반씩 겹친다. 위치 범위는 0 <= bx, by < 15.
- 후보 (bx+dx, by+dy) 기준 좌표 x' 와 (bx, by) 기준 좌표 x 의 관계: x = x' + 2 dx, y = y' + 2 dy.
- 지도가 4096 px보다 크면(UAV-VisLoc 긴 구간, seq_*_full) 블록 수는 grid_size(지도)로 구한다. 4096 px면 15 × 15.
"""
import json
import os
from collections import OrderedDict

import cv2
import numpy as np

N_BLOCK = 15
UNITS_PER_BLOCK = 2  # 이웃 블록 사이 간격 (블록 단위)
STRIDE = 256


def crop_block(city, bx, by):
    """도시 지도에서 블록 (bx, by)를 512 px로 자른다."""
    return city[by * 256:by * 256 + 512, bx * 256:bx * 256 + 512]


def grid_size(city):
    """지도 → (가로 블록 수, 세로 블록 수). 4096 px 지도는 (15, 15)."""
    h, w = city.shape[:2]
    return w // STRIDE - 1, h // STRIDE - 1


def neighbors(bx, by, radius, grid=(N_BLOCK, N_BLOCK)):
    """(dx, dy) 목록: 지도 범위 안의 (2 radius + 1)^2 후보. grid = (가로, 세로) 블록 수."""
    nx, ny = grid
    return [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
            if 0 <= bx + dx < nx and 0 <= by + dy < ny]


class TiledMap:
    """타일로 나눠 저장한 큰 지도 (scripts/export_visloc.py --full). city[y0:y1, x0:x1]과 shape만 지원한다.

    긴 구간 지도는 한 장이 최대 약 40k × 35k px라 통째로 읽으면 메모리가 모자란다. 필요한 타일만 읽고
    최근 max_tiles장만 들고 있는다. 폴더: map.json(width, height, tile) + r{행}_c{열}.jpg.
    """

    def __init__(self, folder, max_tiles=16):   # 2048 px 타일 16장 = 약 200 MB. 3×3 재탐색 범위(1,024 px)는 타일 4장 안
        meta = json.load(open(os.path.join(folder, 'map.json')))
        self.folder, self.tile, self.max_tiles = folder, int(meta['tile']), max_tiles
        self.shape = (int(meta['height']), int(meta['width']), 3)
        self.key = os.path.abspath(folder)   # run_visloc_nav가 지도별 MatchLoc를 나누는 데 쓴다 (md5 대신)
        self._tiles = OrderedDict()

    def _get(self, r, c):
        if (r, c) in self._tiles:
            self._tiles.move_to_end((r, c))
            return self._tiles[(r, c)]
        im = cv2.cvtColor(cv2.imread(os.path.join(self.folder, f'r{r:03d}_c{c:03d}.jpg')), cv2.COLOR_BGR2RGB)
        self._tiles[(r, c)] = im
        if len(self._tiles) > self.max_tiles:
            self._tiles.popitem(last=False)
        return im

    def __getitem__(self, idx):
        ys, xs = idx
        H, W = self.shape[:2]
        y0, y1 = max(ys.start or 0, 0), min(H if ys.stop is None else ys.stop, H)
        x0, x1 = max(xs.start or 0, 0), min(W if xs.stop is None else xs.stop, W)
        out = np.zeros((max(y1 - y0, 0), max(x1 - x0, 0), 3), np.uint8)
        t = self.tile
        for r in range(y0 // t, (y1 - 1) // t + 1 if y1 > y0 else y0 // t):
            for c in range(x0 // t, (x1 - 1) // t + 1 if x1 > x0 else x0 // t):
                im = self._get(r, c)
                ya, yb = max(y0, r * t), min(y1, r * t + im.shape[0])
                xa, xb = max(x0, c * t), min(x1, c * t + im.shape[1])
                out[ya - y0:yb - y0, xa - x0:xb - x0] = im[ya - r * t:yb - r * t, xa - c * t:xb - c * t]
        return out


def load_map(data, st):
    """구간 지도: maps/<set>.jpg(4096 px 한 장)면 배열, maps/<set>/(타일)이면 TiledMap."""
    folder = os.path.join(data, 'maps', st)
    if os.path.isdir(folder):
        return TiledMap(folder)
    return cv2.cvtColor(cv2.imread(folder + '.jpg'), cv2.COLOR_BGR2RGB)
