"""Read public official ZIP index and selected members through bounded HTTP ranges.

No access-control bypass, no full-archive download, no LoMa inference.
"""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

from runtime_paths import AUDIT as ROOT
URL='https://drive.usercontent.google.com/download?id=1L-QdxmXQyfkX2pZPkFc-NnhCkNZdphhN&export=download&confirm=t'
SIZE=4966716421

class Remote(io.RawIOBase):
    def __init__(self):self.pos=0;self.transferred=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=SIZE-self.pos
        n=min(n,SIZE-self.pos)
        if n<=0:return b''
        if n>40*1024**2:raise ValueError('Unexpected large read')
        request=urllib.request.Request(URL,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'})
        with urllib.request.urlopen(request,timeout=60) as response:
            if response.status!=206:raise RuntimeError(f'Range not respected: {response.status}')
            raw=response.read(n+1)
            if len(raw)!=n:raise ValueError(f'Unexpected byte count {len(raw)} != {n}')
        self.pos+=n;self.transferred+=n
        return raw

if __name__=='__main__':
    remote=Remote()
    with zipfile.ZipFile(remote) as z:
        members=z.infolist()
        report={'url':URL,'archive_bytes':SIZE,'member_count':len(members),
            'metadata_members':[i.filename for i in members if i.filename.lower().endswith(('.txt','.json','.md','.csv','.yaml','.yml')) and not i.filename.endswith('groundtruth.txt')],
            'license_members':[i.filename for i in members if any(s in i.filename.lower() for s in ['license','readme','terms','copyright'])],
            'example_members':[i.filename for i in members[:10]],'bytes_transferred':remote.transferred}
        with (ROOT/'sources/NAT2021_zip_index.json').open('x') as f:json.dump(report,f,indent=2)
        print(json.dumps(report,indent=2)[:12000],flush=True)
