"""파이프라인 ⑥: 후보 블록.

좌표 규칙
- 도시 지도 4096 px. 블록 (bx, by)는 지도의 [by*256, by*256+512) 행, [bx*256, bx*256+512) 열 (x가 열).
- 블록은 256 px(2단위)씩 어긋나 있어 이웃과 절반씩 겹친다. 위치 범위는 0 <= bx, by < 15.
- 후보 (bx+dx, by+dy) 기준 좌표 x' 와 (bx, by) 기준 좌표 x 의 관계: x = x' + 2 dx, y = y' + 2 dy.
"""
N_BLOCK = 15
UNITS_PER_BLOCK = 2  # 이웃 블록 사이 간격 (블록 단위)


def crop_block(city, bx, by):
    """도시 지도에서 블록 (bx, by)를 512 px로 자른다."""
    return city[by * 256:by * 256 + 512, bx * 256:bx * 256 + 512]


def neighbors(bx, by, radius):
    """(dx, dy) 목록: 지도 범위 안의 (2 radius + 1)^2 후보."""
    return [(dx, dy) for dx in range(-radius, radius + 1) for dy in range(-radius, radius + 1)
            if 0 <= bx + dx < N_BLOCK and 0 <= by + dy < N_BLOCK]
