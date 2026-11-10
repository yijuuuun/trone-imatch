"""Create fixed-gap40 same-sequence pairs; reuse existing authors' original JPEGs."""
import concurrent.futures
import hashlib
import json
import struct
import sys
import urllib.request
import zipfile
import zlib
from pathlib import Path

from runtime_paths import ROOT as PROJECT
AUDIT=PROJECT/'experiments/2026-11-13_lowlight/calibration_audit'
OUT=PROJECT/'experiments/2026-11-13_lowlight/nat2021_frame_matching_v1'
sys.path.insert(0,str(AUDIT))
from read_nat_index import Remote,URL

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def fetch(info):
    dest=OUT/'frames'/info.filename.split('/')[-2]/Path(info.filename).name
    if dest.exists():raise FileExistsError(dest)
    request=urllib.request.Request(URL,headers={'Range':f'bytes={info.header_offset}-{info.header_offset+info.compress_size+1023}'})
    with urllib.request.urlopen(request,timeout=60) as response:
        if response.status!=206:raise RuntimeError('Public server did not honor Range')
        raw=response.read(info.compress_size+1025)
    if len(raw)>info.compress_size+1024:raise RuntimeError('Response exceeded bound')
    h=struct.unpack('<4s5H3I2H',raw[:30]);start=30+h[-2]+h[-1]
    compressed=raw[start:start+info.compress_size]
    if h[0]!=b'PK\x03\x04' or len(compressed)!=info.compress_size:raise ValueError('ZIP header invalid')
    image=compressed if info.compress_type==0 else zlib.decompress(compressed,-15)
    if len(image)!=info.file_size or zlib.crc32(image)!=info.CRC:raise ValueError('Original member CRC mismatch')
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(image)
    return dict(member=info.filename,path=str(dest.relative_to(PROJECT)),sha256=sha(dest),crc32=f'{info.CRC:08x}',bytes=len(image))

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    protected=[]
    for folder in [PROJECT/'runs',PROJECT/'experiments',PROJECT/'meeting_2026-11-11_v1']:
        protected.extend(p for p in folder.rglob('*') if p.is_file() and not p.is_relative_to(OUT))
    protected+=list((PROJECT/'standalone').glob('*.py'))+[PROJECT/'WEEK2_REPORT.md']
    with (OUT/'PRESERVED_RESULTS_SHA256.json').open('x') as f:json.dump({str(p.relative_to(PROJECT)):sha(p) for p in protected if p.is_file()},f,indent=2)
    previous=json.loads((AUDIT/'NAT2021_SAMPLE_MANIFEST.json').read_text())
    existing={r['member']:r for r in previous['records']}
    remote=Remote()
    with zipfile.ZipFile(remote) as z:
        all_members=z.infolist();pairs=[];wanted=[]
        for sequence in previous['selected_sequences']:
            frames=sorted([i for i in all_members if i.filename.startswith(f'NAT2021/data_seq/{sequence}/') and i.filename.endswith('.jpg')],key=lambda i:i.filename)
            for anchor_name,anchor in [('first',0),('middle',(len(frames)-1)//2)]:
                a,b=frames[anchor],frames[anchor+5]
                assert int(Path(b.filename).stem)-int(Path(a.filename).stem)==5
                old=existing[a.filename];path=AUDIT/old['relative_path'];assert sha(path)==old['sha256']
                pairs.append(dict(pair_id=f'{sequence}_{anchor_name}',sequence=sequence,anchor=anchor_name,
                    gap_frames=5,nominal_gap_seconds=5/30,a_member=a.filename,b_member=b.filename,
                    a_path=str(path.relative_to(PROJECT)),a_sha256=old['sha256']))
                wanted.append(b)
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            records=list(pool.map(fetch,wanted))
    lookup={r['member']:r for r in records}
    for pair in pairs:
        row=lookup[pair['b_member']];pair.update(b_path=row['path'],b_sha256=row['sha256'],b_crc32=row['crc32'])
    manifest=dict(dataset='author-linked public NAT2021-test RGB frames',task='same-sequence temporal frame matching; not geolocalization',
        source_url=URL,n_pairs=40,n_sequences=20,pairs_per_sequence=2,
        sampling='reuse previous fixed20 sequence IDs; first and middle anchors, partner+5 frames; no output-based filtering',
        gap_frames=5,overlap='short fixed gap/full native FoV; actual overlap not GT-known; record post-hoc H proxy without filtering',
        license_note='original-video grant not established in prior audit; local research measurements, no redistribution',
        success_10m_available=False,pairs=pairs)
    with (OUT/'PAIR_MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps({k:v for k,v in manifest.items() if k!='pairs'},indent=2),flush=True)
