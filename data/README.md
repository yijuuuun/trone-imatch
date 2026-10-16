# 데이터

git에는 올리지 않는다(약 340 MB). `data/im_t1_3d_1000/`에 아래 구조로 둔다.

```
data/im_t1_3d_1000/
├── samples.csv          set, k, row, x_norm, y_norm, theta, ...  (정답: 블록 단위 위치, heading 도)
├── 3d_ho38bc/           SF:        k000_uav.jpg (256 px), k000_block.jpg (512 px, 2x2 타일) ... k999
└── 3d_ho37bc/           Singapore: 같은 형식
```

## 받는 법
```bash
gh auth login                     # 처음 한 번 (저장소가 비공개라 필요)
python scripts/download_data.py   # Release data-v1에서 받아 data/im_t1_3d_1000/ 에 푼다
```
브라우저로 받으려면 저장소의 Releases → `data-v1` → `im_t1_3d_1000.zip`을 받아 `data/`에 두고 위 스크립트를 실행한다.

Kaggle에서 돌리려면 이 zip을 자기 Kaggle 데이터셋으로 올리고 `build_kernel.py --dataset`에 그 이름을 넣는다.

## 처음부터 다시 만들기 (보통은 필요 없음)
capstone1 저장소와 Bearing-UAV-90K 원본이 있어야 한다.
```bash
cd capstone1 && python <이 저장소>/scripts/export_from_capstone1.py --n 1000 --out <폴더>
```
행 번호(row)는 capstone1 `diag/harness.py`의 T1 순서와 같다. 그래서 results/의 기준선과 샘플이 1:1로 맞는다.
