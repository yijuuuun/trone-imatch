"""Trone 이미지 매칭 파이프라인: 드론 영상 + 지도 블록 → 위치·heading (학습 없음).

  geometry.py  ④⑤ MAGSAC++ 검증, 호모그래피 → 위치·heading
  matchers.py  ②③ 특징 추출·매칭 (LoMa, RoMa v2, SIFT ...)
  pipeline.py  ①~⑤ 한 샘플 처리 (회전 탐색 포함)
  metrics.py   팀 공통 평가 규칙
"""
def __getattr__(name):
    # CPU evaluation should not require GPU/image-processing dependencies.
    if name == 'get_matcher':
        from .matchers import get_matcher
        return get_matcher
    if name == 'localize':
        from .pipeline import localize
        return localize
    raise AttributeError(name)
