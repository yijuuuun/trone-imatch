#!/usr/bin/env python
"""Self-contained UAV-VisLoc Sequence 03 exporter for Trone-iMatch.

This file intentionally embeds the helper functions formerly imported from
export_external_v3.py, so it can run without that separate module.

The conversion remains provisional:
- UAV GSD is an estimate (default 0.1128 m/px).
- Heading is estimated as -Phi1 + 6 degrees.
- GPS/image timestamp alignment and absolute GPS accuracy are not validated.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tifffile

TARGET_M_PER_PX = 0.25
WIN = 700


def read_geotiff_tags(page):
    """Return raster-to-geographic tiepoint and pixel scale (degrees/px)."""
    tags = page.tags
    if 33550 not in tags or 33922 not in tags:
        raise RuntimeError("GeoTIFF is missing ModelPixelScaleTag or ModelTiepointTag.")
    scale = tags[33550].value
    tie = tags[33922].value
    ri, rj, _rk, mx, my, _mz = tie[:6]
    sx, sy = float(scale[0]), float(scale[1])
    if sx <= 0 or sy <= 0:
        raise RuntimeError(f"Unexpected pixel scale: {scale}")
    return float(ri), float(rj), float(mx), float(my), sx, sy


def source_px_size_m(georef, lat_deg):
    """Ground size (m) of one source pixel: east-west, north-south."""
    _ri, _rj, _mx, _my, sx, sy = georef
    p = math.radians(float(lat_deg))
    m_per_deg_lat = (
        111132.92 - 559.82 * math.cos(2 * p) + 1.175 * math.cos(4 * p)
    )
    m_per_deg_lon = 111412.84 * math.cos(p) - 93.5 * math.cos(3 * p)
    return sx * m_per_deg_lon, sy * m_per_deg_lat


def geo_to_pixel_float(lon, lat, georef):
    """Continuous source pixel coordinate of a GPS point."""
    ri, rj, mx, my, sx, sy = georef
    x = ri + (float(lon) - mx) / sx
    y = rj + (my - float(lat)) / sy
    return x, y


def read_tiled_crop(tif_path: Path, x0: int, y0: int, width: int, height: int):
    """Read only intersecting tiles from an uncompressed chunky RGB tiled TIFF."""
    with tifffile.TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        if not page.is_tiled:
            raise RuntimeError("Expected tiled GeoTIFF.")
        if int(page.compression) != 1:
            raise RuntimeError(f"Expected uncompressed TIFF, got {page.compression}.")
        if int(page.samplesperpixel) != 3 or int(page.bitspersample) != 8:
            raise RuntimeError("Expected 8-bit RGB.")
        if int(page.planarconfig) != 1:
            raise RuntimeError("Expected chunky RGB (PlanarConfiguration=1).")

        image_h, image_w = int(page.imagelength), int(page.imagewidth)
        tile_h, tile_w = int(page.tilelength), int(page.tilewidth)
        if (tile_h, tile_w) != (256, 256):
            raise RuntimeError(f"Expected 256x256 tiles; got {tile_w}x{tile_h}.")
        if x0 < 0 or y0 < 0 or x0 + width > image_w or y0 + height > image_h:
            raise ValueError("Crop outside map.")

        tiles_across = math.ceil(image_w / tile_w)
        out = np.empty((height, width, 3), dtype=np.uint8)
        tx0, tx1 = x0 // tile_w, (x0 + width - 1) // tile_w
        ty0, ty1 = y0 // tile_h, (y0 + height - 1) // tile_h

        with open(tif_path, "rb") as fh:
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    idx = ty * tiles_across + tx
                    offset = int(page.dataoffsets[idx])
                    nbytes = int(page.databytecounts[idx])
                    if nbytes != tile_h * tile_w * 3:
                        raise RuntimeError(f"Unexpected tile byte count at ({tx},{ty}).")
                    fh.seek(offset)
                    raw = fh.read(nbytes)
                    if len(raw) != nbytes:
                        raise IOError(f"Could not read tile ({tx},{ty}).")
                    tile = np.frombuffer(raw, dtype=np.uint8).reshape(tile_h, tile_w, 3)

                    tile_x0, tile_y0 = tx * tile_w, ty * tile_h
                    gx0, gy0 = max(x0, tile_x0), max(y0, tile_y0)
                    gx1 = min(x0 + width, tile_x0 + tile_w)
                    gy1 = min(y0 + height, tile_y0 + tile_h)
                    out[gy0-y0:gy1-y0, gx0-x0:gx1-x0] = tile[
                        gy0-tile_y0:gy1-tile_y0, gx0-tile_x0:gx1-tile_x0
                    ]
        return out


def make_block(window, x0, y0, cx_f, cy_f, px_e, px_n, size=512):
    """Resample to an isotropic 0.25 m/px north-up block centred on GPS."""
    ax, ay = px_e / TARGET_M_PER_PX, px_n / TARGET_M_PER_PX
    c = (size - 1) / 2.0
    matrix = np.array(
        [[ax, 0.0, c - ax * (cx_f - x0)],
         [0.0, ay, c - ay * (cy_f - y0)]],
        dtype=np.float64,
    )
    return cv2.warpAffine(
        window, matrix, (size, size), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def select_even_rows(rows, n):
    if len(rows) < n:
        raise ValueError(f"CSV has only {len(rows)} rows; requested {n}.")
    if n < 2:
        raise ValueError("--n must be at least 2.")
    indices = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    return [(idx, rows[idx]) for idx in indices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="UAV-VisLoc example root containing 03/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=260,
                    help="Evenly spaced candidates; edge samples may be skipped.")
    ap.add_argument("--uav-gsd", type=float, default=0.1128,
                    help="Estimated UAV GSD in metres/pixel.")
    ap.add_argument("--theta-sign", type=float, default=-1.0)
    ap.add_argument("--theta-offset", type=float, default=6.0)
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument("--window-px", type=int, default=WIN)
    ap.add_argument("--provisional", action="store_true",
                    help="Acknowledge the unvalidated scale/heading assumptions.")
    args = ap.parse_args()

    if not args.provisional:
        ap.error("This export is provisional. Re-run with --provisional after reading the caveats.")
    if args.n < 2:
        ap.error("--n must be at least 2.")
    if args.uav_gsd <= 0:
        ap.error("--uav-gsd must be positive.")
    if args.block_px != 512:
        ap.error("This exporter targets 512x512 satellite blocks.")
    if args.window_px <= 0:
        ap.error("--window-px must be positive.")
    if args.theta_sign not in (-1.0, 1.0):
        ap.error("--theta-sign must be +1 or -1.")

    crop = 2 * round(256 * TARGET_M_PER_PX / args.uav_gsd / 2)
    root = Path(args.root)
    seq = root / "03"
    drone_dir, csv_path, tif_path = seq / "drone", seq / "03.csv", seq / "satellite03.tif"
    for path in (csv_path, tif_path, drone_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    selected = select_even_rows(rows, args.n)

    out_root = Path(args.out)
    set_dir = out_root / "uavvisloc_03"
    set_dir.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        georef = read_geotiff_tags(page)
        map_w, map_h = int(page.imagewidth), int(page.imagelength)

    records, skipped = [], []
    half_win = args.window_px // 2
    for source_row_index, row in selected:
        filename = row["filename"]
        uav_path = drone_dir / filename
        if not uav_path.exists():
            raise FileNotFoundError(uav_path)

        cx_f, cy_f = geo_to_pixel_float(row["lon"], row["lat"], georef)
        x0 = int(math.floor(cx_f)) - half_win
        y0 = int(math.floor(cy_f)) - half_win
        if (x0 < 0 or y0 < 0 or x0 + args.window_px > map_w
                or y0 + args.window_px > map_h):
            skipped.append(filename)
            continue

        px_e, px_n = source_px_size_m(georef, row["lat"])
        window = read_tiled_crop(tif_path, x0, y0, args.window_px, args.window_px)
        block = make_block(window, x0, y0, cx_f, cy_f, px_e, px_n, args.block_px)

        with Image.open(uav_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if crop > min(w, h):
                raise ValueError(f"{filename}: crop {crop}px exceeds image {w}x{h}.")
            left, top = (w - crop) // 2, (h - crop) // 2
            uav = im.crop((left, top, left + crop, top + crop))
            uav = uav.resize((256, 256), Image.Resampling.LANCZOS)

        k = len(records) + 1
        stem = f"k{k:03d}"
        uav.save(set_dir / f"{stem}_uav.jpg", quality=95)
        Image.fromarray(block, mode="RGB").save(set_dir / f"{stem}_block.jpg", quality=95)

        theta = (args.theta_sign * float(row["Phi1"]) + args.theta_offset) % 360.0
        records.append({
            "set": "uavvisloc_03", "k": k, "row": source_row_index,
            "x_norm": 0.0, "y_norm": 0.0, "theta": theta,
            "source_filename": filename, "lat": row["lat"], "lon": row["lon"],
            "height": row["height"], "Phi1": row["Phi1"], "Phi2": row["Phi2"],
            "src_px_e_m": round(px_e, 5), "src_px_n_m": round(px_n, 5),
            "uav_crop_px": crop,
        })
        if k == 1 or k % 20 == 0:
            print(f"[{k:03d}] {filename} theta={theta:.2f}")

    if not records:
        raise RuntimeError("No samples exported; all candidates may be outside the map.")

    csv_out = out_root / "samples.csv"
    with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    note = out_root / "UAV_VISLOC_PROVISIONAL_README.txt"
    note.write_text(
        f"""PROVISIONAL UAV-VisLoc export (self-contained exporter)
======================================================
Exported: {len(records)} of {args.n} candidates; skipped at map edge: {len(skipped)}
UAV: central {crop}x{crop}px crop resized to 256x256
Satellite: GPS-centred {args.block_px}x{args.block_px}px at {TARGET_M_PER_PX} m/px
Heading: theta = ({args.theta_sign}) * Phi1 + ({args.theta_offset}) degrees

LIMITATIONS
-----------
1. UAV GSD ({args.uav_gsd} m/px) is estimated, not camera-calibrated.
2. Heading convention is provisional and must be independently validated.
3. GPS accuracy and image/GPS timestamp alignment are unchecked.
4. Do not describe these outputs as a validated external benchmark.
""", encoding="utf-8"
    )
    print(f"EXPORT: {len(records)} exported, {len(skipped)} skipped; crop={crop}px")
    print("samples.csv:", csv_out.resolve())
    print("image folder:", set_dir.resolve())
    if skipped:
        print("skipped:", ", ".join(skipped))


if __name__ == "__main__":
    main()
