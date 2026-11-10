"""Prepare a fresh external workspace; never copy datasets/weights into Git."""
import argparse,hashlib,json,shutil,os
from pathlib import Path
from runtime_paths import ROOT,PACKAGE,AUDIT
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--loma-source',type=Path,required=True)
 p.add_argument('--weights-dir',type=Path,required=True)
 p.add_argument('--bearing-zip',type=Path)
 a=p.parse_args()
 if ROOT.exists():raise FileExistsError('Use a NEW WEEK2_WORKSPACE')
 expected=json.loads((PACKAGE/'configs/loma_source_sha256.json').read_text())
 for name,h in expected.items():
  if hashlib.sha256((a.loma_source/name).read_bytes()).hexdigest()!=h:raise ValueError('LoMa source mismatch: '+name)
 spec=json.loads((PACKAGE/'configs/weights.expected.json').read_text())
 for name,s in spec.items():
  with (a.weights_dir/name).open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
  if h!=s['sha256']:raise ValueError('Weight mismatch: '+name)
 ROOT.mkdir(parents=True);(ROOT/'vendor').mkdir();(ROOT/'downloads').mkdir()
 (ROOT/'vendor/LoMa-main').symlink_to(a.loma_source.resolve(),target_is_directory=True)
 (ROOT/'downloads/weights').symlink_to(a.weights_dir.resolve(),target_is_directory=True)
 if a.bearing_zip:(ROOT/'downloads/Bearing_UAV_90K.zip').symlink_to(a.bearing_zip.resolve())
 (ROOT/'archive_reference/configs').mkdir(parents=True)
 shutil.copyfile(PACKAGE/'configs/weights.expected.json',ROOT/'archive_reference/configs/weights.expected.json')
 shutil.copytree(PACKAGE/'configs/dev100',ROOT/'archive_reference/datasets/singapore')
 AUDIT.mkdir(parents=True)
 (ROOT/'WEEK2_REPORT.md').write_text('Fresh reproduction workspace; original measurements remain in repository results.\n')
 print(ROOT)
if __name__=='__main__':main()
