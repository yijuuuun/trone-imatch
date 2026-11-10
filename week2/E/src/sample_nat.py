"""Deterministic descriptive sample from the authors' public NAT2021 test ZIP.

No LoMa evaluation, no synthetic-parameter fitting, no redistribution step.
Dataset-specific license text was not found in inspected distribution metadata;
the code's Apache license must not be asserted as a confirmed video license.
"""
import concurrent.futures
import hashlib
import json
import struct
import urllib.request
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path
from read_nat_index import Remote,URL,ROOT

def fetch(info,out):
    request=urllib.request.Request(URL,headers={'Range':f'bytes={info.header_offset}-{info.header_offset+info.compress_size+1023}'})
    with urllib.request.urlopen(request,timeout=60) as response:
        if response.status!=206:raise ValueError('Range not respected')
        raw=response.read(info.compress_size+1025)
    if len(raw)>info.compress_size+1024:raise ValueError('Response too large')
    header=struct.unpack('<4s5H3I2H',raw[:30])
    if header[0]!=b'PK\x03\x04':raise ValueError('Invalid local ZIP header')
    start=30+header[-2]+header[-1]
    compressed=raw[start:start+info.compress_size]
    if len(compressed)!=info.compress_size:raise ValueError('Local ZIP extra header exceeds bound')
    decoded=compressed if info.compress_type==0 else zlib.decompress(compressed,-15)
    if len(decoded)!=info.file_size or zlib.crc32(decoded)!=info.CRC:raise ValueError('Original member CRC mismatch')
    target=out/info.filename.split('/')[-2]/Path(info.filename).name
    target.parent.mkdir(exist_ok=True)
    with target.open('xb') as f:f.write(decoded)
    return dict(member=info.filename,relative_path=str(target.relative_to(ROOT)),bytes=len(decoded),
                crc32=f'{info.CRC:08x}',sha256=hashlib.sha256(decoded).hexdigest())

if __name__=='__main__':
    out=ROOT/'nat2021_sample';out.mkdir(exist_ok=False)
    remote=Remote()
    with zipfile.ZipFile(remote) as z:
        sequences=defaultdict(list)
        for info in z.infolist():
            parts=info.filename.split('/')
            if len(parts)==4 and parts[:2]==['NAT2021','data_seq'] and info.filename.lower().endswith(('.jpg','.png')):
                sequences[parts[2]].append(info)
        names=sorted(sequences)
        chosen=[names[round(i*(len(names)-1)/19)] for i in range(20)]
        selected=[]
        for name in chosen:
            frames=sorted(sequences[name],key=lambda i:i.filename)
            for idx in [0,(len(frames)-1)//2,len(frames)-1]:selected.append(frames[idx])
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            records=list(pool.map(lambda info:fetch(info,out),selected))
        report=dict(source_url=URL,split='NAT2021-test; external descriptive analysis only, no LoMa tuning',
            archive_bytes=4966716421,population_sequences=len(names),population_frames=sum(len(v) for v in sequences.values()),
            sampling='sort sequence IDs; 20 indices round(i*(N-1)/19); first, floor((n-1)/2), last frame per sequence',
            selected_sequences=chosen,sample_frames=len(records),records=records,
            license_status='Author-linked public download; no dataset-specific license text found. No redistribution or code-license transfer asserted.',
            official_scene_labels_available=False,official_LAI_labels_used=False,parameter_fit=False)
        with (ROOT/'NAT2021_SAMPLE_MANIFEST.json').open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2),flush=True)
