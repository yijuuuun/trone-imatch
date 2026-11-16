from pathlib import Path
import numpy as np
import torch
from hashlib import sha256 as _sha256
def sha256(path):
    h=_sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()

class BearingAdapter:
    def __init__(self, repo, weights_dir, device):
        import sys
        sys.path.insert(0, str(repo))
        # Official utils_transform accesses np.complex_ during import on NumPy 2.
        for name, value in {'complex_': np.complex128, 'float_': np.float64}.items():
            if name not in np.__dict__:
                setattr(np, name, value)
        # imgaug 0.4 imports np.sctypes, removed in NumPy 2. This compatibility alias
        # affects import-time augmentation setup; evaluation uses only ToTensor/Normalize.
        if not hasattr(np, 'sctypes'):
            np.sctypes = {'int': [np.int8, np.int16, np.int32, np.int64],
                          'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
                          'float': [np.float16, np.float32, np.float64],
                          'complex': [np.complex64, np.complex128], 'others': [bool, object, str, bytes]}
        from cvphr.models.posaglreg.models import load_config_and_model
        cls, kw = load_config_and_model(str(weights_dir))
        self.model = cls(**kw)
        path = Path(weights_dir) / 'best_model.pth'
        state = torch.load(path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model_state_dict'], strict=True)
        self.model = self.model.float().to(device).eval()
        self.device, self.modules = device, [self.model]
        self.metadata = {'model': 'Bearing-UAV', 'variant': 'official cross_view',
                         'weights': str(path), 'checkpoint_sha256': sha256(path),
                         'training_regime': 'public-pretrained: training-city overlap must be audited'}

    def predict(self, uav, block):
        # Official p1,p2,p3,p4 order = TL,TR,BL,BR; fifth image = UAV.
        patches = [block[y:y+256, x:x+256] for x, y in [(0, 0), (256, 0), (0, 256), (256, 256)]] + [uav]
        x = np.stack(patches).astype(np.float32) / 255
        x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        x = torch.from_numpy(x.transpose(0, 3, 1, 2).copy())[None].to(self.device)
        pos, _direction = self.model(x)
        xy = pos[0].float().cpu().numpy()
        return {'x': float(xy[0]), 'y': float(xy[1]), 'valid': True}
