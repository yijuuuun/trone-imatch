"""Restore only archived development k0..99; preserve source identities and hashes."""
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

from runtime_paths import ROOT
if __name__=='__main__':
    archive=ROOT/'downloads/Bearing_UAV_90K.zip'
    out=ROOT/'data/singapore_dev100'
    subprocess.run([sys.executable,str(Path(__file__).with_name('restore_singapore.py')),'--zip',str(archive),'--out',str(out),'--limit','100'],check=True)
    rows=list(csv.DictReader((ROOT/'archive_reference/datasets/singapore/source_mapping.csv').open()))[:100]
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if n.startswith('Bearing_UAV_90K/city_rsi/') and n.endswith('.json')]
        metadata={name:json.loads(z.read(name)) for name in names}
        (out/'rsi_metadata.json').write_text(json.dumps(metadata,indent=2))
        selected={}
        for row in rows:
            name=row['target_path'].replace('.jpg','.json')
            if name in z.namelist():selected[row['k']]=json.loads(z.read(name))
        (out/'uav_metadata.json').write_text(json.dumps(selected,indent=2))
    scenes=Counter(tuple(r[f'p{i}_path'] for i in range(1,5)) for r in rows)
    paths=Counter(r['target_path'] for r in rows)
    image_hashes=Counter(hashlib.sha256((out/'3d_ho37bc'/f'k{k:03d}_uav.jpg').read_bytes()).hexdigest() for k in range(100))
    report={'n':100,'k_range':[0,99],'unique_uav_paths':len(paths),'unique_uav_jpeg_hashes':len(image_hashes),
       'unique_satellite_scenes':len(scenes),'repeated_uav_paths':sum(n-1 for n in paths.values()),
       'repeated_satellite_scene_samples':sum(n-1 for n in scenes.values()),
       'test_images_restored':0,'map_condition':'Oracle T1: archived GT-containing four-tile block',
       'team_current_split_parity':'not verified without team source',
       'scene_id_by_k':{r['k']:r['p1_path'].rsplit('_base_',1)[0] for r in rows}}
    (out/'duplicate_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='scene_id_by_k'},indent=2))
