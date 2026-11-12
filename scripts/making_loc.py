0. 환경 준비 (한 번만)

import sys, tarfile, numpy as np, pandas as pd, torch, cv2
from pathlib import Path
from scipy.spatial import cKDTree

# imgaug가 NumPy 2.x에서 import되도록 임시 패치 (cvphr import 전에 실행)
np.sctypes = {"int":[np.int8,np.int16,np.int32,np.int64],
              "uint":[np.uint8,np.uint16,np.uint32,np.uint64],
              "float":[np.float16,np.float32,np.float64],
              "complex":[np.complex64,np.complex128],
              "others":[bool,object,str,bytes]}
np.complex = np.complex128; np.bool = np.bool_

REPO   = Path("/kaggle/input/datasets/jimin562/capstone1/capstone1-main/capstone1-main")
C5     = Path("/kaggle/input/datasets/jimin562/c5-trone-imatch-evaluation")
DATA   = Path("/kaggle/input/datasets/jimin562/bearing-uav-90k-city-archives")
META   = DATA/"Bearing_UAV_90K/c4m_254k_96bc_b15_s100_v3d/metadata/metadata.csv"
B1_TGZ = "/kaggle/input/datasets/jimin562/capstone1/b1_retrained.tgz"
CORR   = REPO/"results/corr"

for p in (str(REPO), str(C5)):
    sys.path.insert(0, p)

import os; os.chdir(REPO)   # diag 모듈이 os.getcwd()를 sys.path에 넣음

from diag.diag_modules import DEV, to_tensor, enc, head
from cvphr.models.posaglreg.models import load_config_and_model
from cvphr.models.corrmatch import CorrMatchNet, north_up
from imatch1.nav import make_traj, nav_run, ODO, Nearest3D

1. 데이터 어댑터 (B1·Corr 공용)


class B1Data3D:
    def __init__(self, metadata, city_code, base_dir):
        self.base = Path(base_dir)
        self.df = metadata[metadata["p1_path"].astype(str).str.contains(city_code, regex=False)] \
                    .copy().reset_index(drop=True)
        self.df["px"] = self.df.block_x*256 + 256 + self.df.x_norm*128
        self.df["py"] = self.df.block_y*256 + 256 + self.df.y_norm*128
        self.xy = self.df[["px","py"]].to_numpy(np.float32)
        self.tree = cKDTree(self.xy)
        self.tile_cache, self.target_cache = {}, {}

    def target(self, row):
        p = str(row["target_path"])
        if p not in self.target_cache:
            self.target_cache[p] = cv2.cvtColor(cv2.imread(str(self.base/p)), cv2.COLOR_BGR2RGB)
        return self.target_cache[p]

    def tiles(self, row):   # row: 일반 행 또는 (block_x, block_y)가 index인 행
        if "block_x" in row.index: key = (int(row["block_x"]), int(row["block_y"]))
        else:                      key = (int(row.name[0]), int(row.name[1]))
        if key not in self.tile_cache:
            self.tile_cache[key] = [
                cv2.cvtColor(cv2.imread(str(self.base/row[c])), cv2.COLOR_BGR2RGB)
                for c in ("p1_path","p2_path","p3_path","p4_path")]   # TL, TR, BL, BR
        return self.tile_cache[key]


2. B1 loc

class B1Loc:
    def __init__(self, model, data):
        self.m, self.data = model, data
        self.tile_feature_cache = {}
        self.block_rows = data.df.drop_duplicates(["block_x","block_y"]).set_index(["block_x","block_y"])

    def block_features(self, bx, by):
        key = (int(bx), int(by))
        if key not in self.tile_feature_cache:
            tiles = self.data.tiles(self.block_rows.loc[key])
            x = torch.stack([to_tensor(im) for im in tiles]).to(DEV)
            with torch.no_grad():
                self.tile_feature_cache[key] = enc(self.m, x).detach()   # [4, 1024]
        return self.tile_feature_cache[key]

    def __call__(self, crops, headings, blocks):    # headings는 무시 (B1은 나침반 미사용)
        uav = torch.stack([to_tensor(c) for c in crops]).to(DEV)
        nb  = torch.stack([self.block_features(bx, by) for bx, by in blocks])   # [B,4,1024]
        with torch.no_grad():
            fi  = enc(self.m, uav)
            out = head(self.m, fi, nb)       # (pos, dir, attn, psg, ctx, ce)
        self.last_direction = out[1].cpu().numpy()
        return out[0].cpu().numpy()          # 블록 중심 기준, 1 = 128px


3. CorrMatchNet loc

class CorrNavLoc:
    def __init__(self, model, data, block_rows):
        self.m, self.data, self.block_rows = model, data, block_rows
        self.cache = {}

    @torch.no_grad()
    def __call__(self, crops, headings, blocks):
        tf = []
        for bx, by in blocks:
            key = (int(bx), int(by))
            if key not in self.cache:
                tiles = self.data.tiles(self.block_rows.loc[key])
                x = torch.stack([to_tensor(t) for t in tiles]).to(DEV)
                self.cache[key] = self.m.tile_features(x).detach()   # [4,512,32,32]
            tf.append(self.cache[key])
        tf = torch.stack(tf)                                          # [B,4,512,32,32]
        u = torch.stack([to_tensor(north_up(c, float(h)))             # 북쪽 정렬 + 중앙 176px
                         for c, h in zip(crops, headings)]).to(DEV)
        pos, conf = self.m.predict(u, tf)                             # corr_config.json의 decode 사용
        return pos.cpu().numpy()


4. 모델 로드와 loc 조립

md = pd.read_csv(META)

def extract_b1(member_dir, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(B1_TGZ, "r:gz") as tar:
        for f in ("best_model.pth", "training_configure.json"):
            (out/f).write_bytes(tar.extractfile(f"{member_dir}/{f}").read())
    return out

def load_b1(d):
    cls, kw = load_config_and_model(str(d))
    m = cls(**kw)
    m.load_state_dict(torch.load(d/"best_model.pth", map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    return m.to(DEV).eval()

CITY = {  # city: (코드, B1 tgz 폴더, Corr 폴더)
 "SG": ("37bc", "c4ma/phr5_d96100_3d_ho37bc_b16_l0.5_e100_gNone_20261005_152953", "p2bv4_3d_ho37bc_20261013_173659"),
 "SF": ("38bc", "c4ma/phr5_d96100_3d_ho38bc_b16_l0.5_e100_gNone_20261004_225155", "p2bv4_3d_ho38bc_20261013_135730"),
}

def build(city):
    code, b1_member, corr_dir = CITY[city]
    data   = B1Data3D(md, code, DATA)
    b1_loc = B1Loc(load_b1(extract_b1(b1_member, f"/kaggle/working/b1_{city}")), data)
    corr_m = CorrMatchNet.load(CORR/corr_dir, device=DEV)
    corr_loc = CorrNavLoc(corr_m, data, b1_loc.block_rows)
    frames = Nearest3D(str(DATA), f"3d_ho{code}")
    return b1_loc, corr_loc, frames
5. 실행

starts, goals = make_traj(32, 1000, 7)
b1_loc, corr_loc, frames = build("SF")

for name, loc in [("B1", b1_loc), ("Corr", corr_loc)]:
    log = []
    success, reason, med = nav_run(loc, frames, starts, goals, decode_px=128,
                                   odo=ODO["vio"], seed=7, log=log)   # 반환값은 3개
    print(name, success.sum(), "/", len(success), pd.Series(reason).value_counts().to_dict())