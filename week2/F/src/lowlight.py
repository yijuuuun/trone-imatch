"""Optional RGB uint8 UAV preprocessing. Satellite images must bypass this module.

Development synthesis only: these are NOT verified T4R/NAT2021 settings.
No LoMa resize, matching, geometry, or team defaults are changed here.
"""
import argparse
from pathlib import Path
import numpy as np

SYNTHESIS = {'mild': (0.45, 1000.0, 0.008),
             'severe': (0.15, 500.0, 0.015)}


def validate(image):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected RGB uint8 HxWx3')


def synthesize(image, condition='daylight', seed=0):
    """Linear-light exposure loss plus Poisson shot and Gaussian read noise.

    Pass the same sample seed for all variants; generate once before enhancement.
    Intensities/exposure and noise strengths are provisional development choices.
    """
    validate(image)
    if condition == 'daylight':
        return image.copy()
    exposure, photons, read_sigma = SYNTHESIS[condition]
    x = image.astype(np.float64) / 255
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    rng = np.random.default_rng(seed)
    noisy = rng.poisson(linear * exposure * photons) / photons
    noisy += rng.normal(0, read_sigma, x.shape)
    noisy = np.clip(noisy, 0, 1)
    srgb = np.where(noisy <= 0.0031308, 12.92 * noisy,
                    1.055 * noisy ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def enhance(image, method='none', gamma=0.6, clip_limit=2.0, grid=8):
    validate(image)
    if method not in ('none', 'gamma', 'clahe', 'gamma_clahe'):
        raise ValueError(method)
    if gamma <= 0 or clip_limit <= 0 or grid < 1:
        raise ValueError('Positive enhancement parameters required')
    result = image.copy()
    if method in ('gamma', 'gamma_clahe'):
        lut = np.rint((np.arange(256) / 255) ** gamma * 255).astype(np.uint8)
        result = lut[result]
    if method in ('clahe', 'gamma_clahe'):
        import cv2
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit,
                                      tileGridSize=(grid, grid)).apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result


def preprocess_uav(image, lowlight='daylight', enhancement='none', seed=0):
    return enhance(synthesize(image, lowlight, seed), enhancement)


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--lowlight', choices=['daylight', *SYNTHESIS], default='daylight')
    parser.add_argument('--enhance', choices=['none', 'gamma', 'clahe', 'gamma_clahe'], default='none')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error('Output must be a new file')
    bgr = cv2.imread(str(args.input))
    if bgr is None:
        parser.error('Cannot decode input')
    result = preprocess_uav(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.lowlight, args.enhance, args.seed)
    ok, encoded = cv2.imencode(args.output.suffix, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError('Image encoding failed')
    with args.output.open('xb') as stream:
        stream.write(encoded.tobytes())


if __name__ == '__main__':
    main()
