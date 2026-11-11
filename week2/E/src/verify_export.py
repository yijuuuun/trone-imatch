"""CPU-only import and provenance checks; no dataset or GPU inference required."""
import ast,hashlib,importlib,json
from pathlib import Path
from runtime_paths import PACKAGE
def main():
 manifest=json.loads((PACKAGE/'configs/source_manifest.json').read_text())
 for r in manifest:
  p=PACKAGE/r['export_path']
  assert hashlib.sha256(p.read_bytes()).hexdigest()==r['export_sha256'],str(p)
 files=sorted((PACKAGE/'src').glob('*.py'))
 for p in files:ast.parse(p.read_text(),filename=str(p))
 for name in ['lowlight','loma_backend','smoke_gpu','pilot_matrix','analyze_pilot']:
  importlib.import_module(name)
 import numpy as np
 from lowlight import synthesize,enhance
 rgb=np.random.default_rng(0).integers(0,256,(32,32,3),dtype=np.uint8)
 for light in ['daylight','mild','severe']:
  a=synthesize(rgb,light,20261110);b=synthesize(rgb,light,20261110)
  assert np.array_equal(a,b) and a.dtype==np.uint8 and a.shape==rgb.shape
  for method in ['none','gamma','clahe','gamma_clahe']:assert enhance(a,method).shape==rgb.shape
 print(f'PASS: {len(files)} Python files parsed; core imports, source manifest and deterministic image smoke checks')
if __name__=='__main__':main()
