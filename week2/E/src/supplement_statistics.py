"""Supplementary noise proxies and explicitly manual first-frame scene grouping."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

from runtime_paths import AUDIT as ROOT
def write(path,rows):
    with path.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

groups={
    'N01001':'road_urban','N01010':'road_urban','N02009':'road_urban','N02018':'road_urban',
    'N02028':'road_urban','N02037':'road_urban','N02047':'road_urban','N02056':'dark_path_uncertain',
    'N03001':'road_urban','N03011':'road_urban','N04004':'road_urban','N04014':'dark_path_uncertain',
    'N04023':'open_paved_or_sports_area','N04032':'open_paved_or_sports_area','N04042':'dark_path_uncertain',
    'N05005':'distant_urban_skyline','N08002':'road_urban','N08011':'road_urban','N08021':'road_urban','N09006':'dark_path_uncertain'}

if __name__=='__main__':
    manifest=json.loads((ROOT/'NAT2021_SAMPLE_MANIFEST.json').read_text());noise=[]
    for r in manifest['records']:
        p=ROOT/r['relative_path'];rgb=cv2.cvtColor(cv2.imread(str(p)),cv2.COLOR_BGR2RGB).astype(np.float64)/255
        y=rgb@np.array([.2126,.7152,.0722])
        local=(y[:-1,:-1]+y[:-1,1:]+y[1:,:-1]+y[1:,1:])/4
        hh=(y[:-1,:-1]-y[:-1,1:]-y[1:,:-1]+y[1:,1:])/2
        v=hh[local<.1]
        noise.append(dict(sequence=p.parent.name,frame=p.name,dark_samples=len(v),
            dark_hh_rms_NOT_sensor_sigma=float(np.sqrt(np.mean(v*v))) if len(v) else None,
            dark_hh_abs_p95=float(np.percentile(np.abs(v),95)) if len(v) else None,
            dark_hh_exact_zero_ratio=float(np.mean(v==0)) if len(v) else None,
            limitation='JPEG texture/compression/ISP/quantization mixed; sRGB-domain native-pixel proxy only'))
    write(ROOT/'NAT2021_NOISE_PROXY_SUPPLEMENT.csv',noise)
    seq=list(csv.DictReader((ROOT/'NAT2021_SEQUENCE_STATISTICS.csv').open()))
    labels=[dict(sequence=s['sequence'],manual_visual_group=groups[s['sequence']],
        label_source='assistant inspection of first-frame montage; not official annotation',
        label_scope='first-frame visual structure applied descriptively to three sampled frames',
        campus_identity='unverified',uncertainty='road/urban groups overlap semantically; dark path labels tentative') for s in seq]
    write(ROOT/'SCENE_VISUAL_LABELS.csv',labels)
    grouped=defaultdict(list)
    for s in seq:grouped[groups[s['sequence']]].append(s)
    out=[]
    for group,records in grouped.items():
        values=[float(s['mean_srgb_code_255']) for s in records]
        out.append(dict(manual_visual_group=group,sequences=len(records),frames=3*len(records),
            mean_sequence_brightness_255=float(np.mean(values)),min_sequence_mean_255=min(values),max_sequence_mean_255=max(values),
            official_scene_category=False,campus_confirmed=False,interpretation='descriptive convenience grouping, not dataset-wide scene statistics'))
    write(ROOT/'SCENE_GROUP_STATISTICS.csv',out)
    print(json.dumps(out,indent=2))
