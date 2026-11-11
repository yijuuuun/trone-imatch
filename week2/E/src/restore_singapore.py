"""Restore only the requested Method-B Singapore pairs from an external ZIP.

Every output JPEG must match the stored reconstruction hash. This does NOT
certify pixel identity to the unavailable original historical harness export.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

from runtime_paths import ROOT
ROOT=ROOT/'archive_reference'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--zip',dest='zip_path',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--limit',type=int,default=100)
    a=p.parse_args()
    if not 1<=a.limit<=100:raise ValueError('Require1..100; original k order is preserved')
    if a.out.exists():raise FileExistsError('Never overwrite an existing dataset')
    import cv2
    import numpy as np
    base=ROOT/'datasets/singapore'
    spec=json.loads((base/'export_provenance.json').read_text())
    with (base/'source_mapping.csv').open(newline='') as f:rows=list(csv.DictReader(f))
    with (base/'samples.csv').open(newline='') as f:samples={int(r['k']):r for r in csv.DictReader(f)}
    assert len(rows)==100 and {int(r['k']) for r in rows}==set(range(100))
    rows=sorted(rows,key=lambda r:int(r['k']))[:a.limit]
    with zipfile.ZipFile(a.zip_path) as z:
        raw=z.read(spec['metadata_member']);assert hashlib.sha256(raw).hexdigest()==spec['metadata_sha256']
        meta=list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
        a.out.mkdir(parents=True);images=a.out/'3d_ho37bc';images.mkdir()
        for r in rows:
            s=meta[int(r['row'])]
            sample=samples[int(r['k'])]
            assert sample['row']==r['row']
            assert all(float(sample[k])==float(s[k]) for k in ['x_norm','y_norm','theta'])
            def image(column):
                name=s[column].removeprefix('./')
                assert name==r[column] and name.startswith('Bearing_UAV_90K/cityc/') and '..' not in name.split('/')
                raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==spec['source_image_sha256'][name]
                im=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
                assert im is not None and im.shape==(256,256,3)
                return im
            patches=[image(f'p{i}_path') for i in range(1,5)]
            block=np.vstack([np.hstack(patches[:2]),np.hstack(patches[2:])])
            for kind,im in [('uav',image('target_path')),('block',block)]:
                ok,b=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,97]);assert ok
                name=f'3d_ho37bc/k{int(r["k"]):03d}_{kind}.jpg'
                assert hashlib.sha256(b.tobytes()).hexdigest()==spec['exported_image_sha256'][name], 'JPEG bytes differ; use recorded OpenCV version and review before benchmarking'
                (a.out/name).write_bytes(b.tobytes())
    # Preserve original metadata bytes rather than generate labels or renumber k.
    shutil.copyfile(base/'samples.csv',a.out/'samples.csv')
    shutil.copyfile(base/'source_mapping.csv',a.out/'source_mapping.csv')
    report={'restored_pairs':a.limit,'k_range':[0,a.limit-1],'source_images_and_output_JPEG_SHA256_verified':True,
            'all_manifest_rows_retained':100,'image_subset_only':a.limit<100,
            'original_harness_pixel_identity_verified':False,'new_GT_or_indices_generated':False,'opencv':cv2.__version__}
    (a.out/'restoration_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
