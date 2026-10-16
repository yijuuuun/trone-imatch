# Kaggle launcher: install vismatch, unpack the sample files, then run imatch_run.main()
import subprocess, sys, glob, os, zipfile, shutil
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vismatch'], check=True)
src = os.path.dirname(glob.glob('/kaggle/input/**/samples.csv', recursive=True)[0])
DATA = '/kaggle/working/data'; os.makedirs(DATA, exist_ok=True)
for f in os.listdir(src):
    p = os.path.join(src, f)
    if f.endswith('.zip'):
        zipfile.ZipFile(p).extractall(os.path.join(DATA, f[:-4]))
    elif os.path.isdir(p):
        shutil.copytree(p, os.path.join(DATA, f), dirs_exist_ok=True)
    else:
        shutil.copy(p, DATA)
# a zip may hold its folder or only the files
for d in glob.glob(DATA + '/*_ho*'):
    inner = os.path.join(d, os.path.basename(d))
    if os.path.isdir(inner):
        for f in os.listdir(inner): shutil.move(os.path.join(inner, f), d)
print('data', sorted(os.listdir(DATA)), len(glob.glob(DATA + '/3d_ho38bc/*')), flush=True)
sys.argv = ['imatch_run', '--data', DATA, '--out', '/kaggle/working/out', '--sets', '3d_ho38bc,3d_ho37bc',
            '--matchers', 'sift-lightglue,loma,romav2', '--scales', '1,2', '--angles', '8']
