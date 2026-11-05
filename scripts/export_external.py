#!/usr/bin/env python
"""
C5 UAV-VisLoc (sequence 03) exporter for Trone-iMatch.

C5 DATA PREPARATION:
- Converts UAV-VisLoc Sequence 03 into the Trone-iMatch external-data format.
- This is a PROVISIONAL data-preparation script, not a validated benchmark converter.
- It uses GPS as the centre of each 512x512 satellite crop (x_norm=y_norm=0).
- It uses Phi1 as a provisional heading and exposes sign/offset switches.
- The camera intrinsics / exact UAV ground footprint are not provided in the local
  example README. Therefore --uav-crop-px is an explicit scale assumption.
- Do not report evaluation metrics as a rigorous external benchmark until the
  crop scale and heading convention have been independently validated.

Output layout: <out>/samples.csv and <out>/uavvisloc_03/k###_{uav,block}.jpg.
The GeoTIFF is read tile-by-tile. The full 2.58 GB image is never loaded.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


def read_geotiff_tags(page):
    """Return raster-to-geographic tiepoint and pixel scale."""
    tags = page.tags
    if 33550 not in tags or 33922 not in tags:
        raise RuntimeError("GeoTIFF is missing ModelPixelScaleTag or ModelTiepointTag.")
    scale = tags[33550].value
    tie = tags[33922].value
    # First tiepoint: raster_i, raster_j, raster_k, model_x, model_y, model_z
    ri, rj, _rk, mx, my, _mz = tie[:6]
    sx, sy = float(scale[0]), float(scale[1])
    if sx <= 0 or sy <= 0:
        raise RuntimeError(f"Unexpected pixel scale: {scale}")
    return float(ri), float(rj), float(mx), float(my), sx, sy


def geo_to_pixel(lon, lat, georef):
    ri, rj, mx, my, sx, sy = georef
    # Dataset GeoTIFF uses geographic coordinates: X=longitude, Y=latitude.
    x = ri + (float(lon) - mx) / sx
    y = rj + (my - float(lat)) / sy
    return int(round(x)), int(round(y))


def read_tiled_crop(tif_path: Path, x0: int, y0: int, width: int, height: int):
    """
    Read a crop from an uncompressed, chunky RGB tiled TIFF using direct tile
    offsets. Only intersecting tiles are read.
    """
    with tifffile.TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        if not page.is_tiled:
            raise RuntimeError("Expected tiled GeoTIFF.")
        if int(page.compression) != 1:
            raise RuntimeError(
                f"Expected uncompressed TIFF (compression=1), got {page.compression}."
            )
        if int(page.samplesperpixel) != 3 or int(page.bitspersample) != 8:
            raise RuntimeError(
                f"Expected 8-bit RGB; got samples={page.samplesperpixel}, "
                f"bits={page.bitspersample}."
            )
        if int(page.planarconfig) != 1:
            raise RuntimeError("Expected chunky/interleaved RGB (PlanarConfiguration=1).")

        image_h, image_w = int(page.imagelength), int(page.imagewidth)
        tile_h, tile_w = int(page.tilelength), int(page.tilewidth)
        if tile_h != 256 or tile_w != 256:
            raise RuntimeError(f"Expected 256x256 tiles; got {tile_w}x{tile_h}.")

        if x0 < 0 or y0 < 0 or x0 + width > image_w or y0 + height > image_h:
            raise ValueError(
                f"Crop outside map: x={x0}:{x0+width}, y={y0}:{y0+height}, "
                f"map={image_w}x{image_h}"
            )

        tiles_across = math.ceil(image_w / tile_w)
        out = np.empty((height, width, 3), dtype=np.uint8)

        tx0, tx1 = x0 // tile_w, (x0 + width - 1) // tile_w
        ty0, ty1 = y0 // tile_h, (y0 + height - 1) // tile_h

        with open(tif_path, "rb") as fh:
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    tile_index = ty * tiles_across + tx
                    offset = int(page.dataoffsets[tile_index])
                    bytecount = int(page.databytecounts[tile_index])
                    expected = tile_h * tile_w * 3
                    if bytecount != expected:
                        raise RuntimeError(
                            f"Unexpected tile byte count at ({tx},{ty}): "
                            f"{bytecount}, expected {expected}."
                        )
                    fh.seek(offset)
                    raw = fh.read(bytecount)
                    if len(raw) != bytecount:
                        raise IOError(f"Could not read complete tile ({tx},{ty}).")
                    tile = np.frombuffer(raw, dtype=np.uint8).reshape(
                        tile_h, tile_w, 3
                    )

                    tile_x0, tile_y0 = tx * tile_w, ty * tile_h
                    gx0, gy0 = max(x0, tile_x0), max(y0, tile_y0)
                    gx1 = min(x0 + width, tile_x0 + tile_w)
                    gy1 = min(y0 + height, tile_y0 + tile_h)

                    out[
                        gy0 - y0 : gy1 - y0,
                        gx0 - x0 : gx1 - x0,
                    ] = tile[
                        gy0 - tile_y0 : gy1 - tile_y0,
                        gx0 - tile_x0 : gx1 - tile_x0,
                    ]

        return out, (image_w, image_h), georef_from_page(page)


def georef_from_page(page):
    return read_geotiff_tags(page)


def select_even_rows(rows, n):
    if len(rows) < n:
        raise ValueError(f"CSV has only {len(rows)} rows; requested {n}.")
    indices = [round(i * (len(rows) - 1) / (n - 1)) for i in range(n)]
    return [(idx, rows[idx]) for idx in indices]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\user\Downloads\UAV_VisLoc_example (1)")
    ap.add_argument("--out", default="data_uavvisloc")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument(
        "--uav-crop-px",
        type=int,
        default=1024,
        help="Central square crop in original UAV pixels before resize to 256.",
    )
    ap.add_argument("--block-px", type=int, default=512)
    ap.add_argument(
        "--theta-sign",
        type=float,
        default=1.0,
        help="Provisional Phi1 multiplier: +1 or -1.",
    )
    ap.add_argument(
        "--theta-offset",
        type=float,
        default=0.0,
        help="Provisional heading offset in degrees.",
    )
    ap.add_argument(
        "--provisional",
        action="store_true",
        help="Required acknowledgement that scale/heading are not yet validated.",
    )
    args = ap.parse_args()

    if not args.provisional:
        ap.error(
            "Export is provisional. Re-run with --provisional only after reading "
            "the scale/heading caveats in this script."
        )
    if args.n < 2:
        ap.error("--n must be at least 2.")
    if args.block_px != 512:
        ap.error("This exporter currently targets 512x512 satellite blocks.")
    if args.uav_crop_px <= 0 or args.uav_crop_px % 2:
        ap.error("--uav-crop-px must be a positive even number.")
    if args.theta_sign not in (-1.0, 1.0):
        ap.error("--theta-sign must be +1 or -1.")

    root = Path(args.root)
    seq = root / "03"
    drone_dir = seq / "drone"
    csv_path = seq / "03.csv"
    tif_path = seq / "satellite03.tif"

    for p in (csv_path, tif_path, drone_dir):
        if not p.exists():
            raise FileNotFoundError(p)

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    selected = select_even_rows(rows, args.n)

    out_root = Path(args.out)
    set_name = "uavvisloc_03"
    set_dir = out_root / set_name
    set_dir.mkdir(parents=True, exist_ok=True)

    # Read GeoTIFF metadata once; only tile pixels intersecting each crop are read.
    with tifffile.TiffFile(str(tif_path)) as tif:
        page = tif.pages[0]
        georef = read_geotiff_tags(page)
        map_w, map_h = int(page.imagewidth), int(page.imagelength)

    records = []
    half_block = args.block_px // 2
    half_uav = args.uav_crop_px // 2

    for k, (source_row_index, row) in enumerate(selected, start=1):
        filename = row["filename"]
        uav_path = drone_dir / filename
        if not uav_path.exists():
            raise FileNotFoundError(uav_path)

        # UAV central square crop, then resize to the required 256x256 input.
        with Image.open(uav_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            if args.uav_crop_px > min(w, h):
                raise ValueError(
                    f"{filename}: crop {args.uav_crop_px}px exceeds image {w}x{h}."
                )
            left = (w - args.uav_crop_px) // 2
            top = (h - args.uav_crop_px) // 2
            uav_crop = im.crop(
                (left, top, left + args.uav_crop_px, top + args.uav_crop_px)
            )
            uav_256 = uav_crop.resize((256, 256), Image.Resampling.LANCZOS)

        # GPS centre -> GeoTIFF pixel -> 512x512 satellite crop.
        cx, cy = geo_to_pixel(row["lon"], row["lat"], georef)
        x0, y0 = cx - half_block, cy - half_block
        if x0 < 0 or y0 < 0 or x0 + args.block_px > map_w or y0 + args.block_px > map_h:
            raise ValueError(
                f"{filename}: GPS-centred map crop falls outside GeoTIFF "
                f"(centre={cx},{cy}; map={map_w}x{map_h})."
            )

        with tifffile.TiffFile(str(tif_path)) as tif:
            block, _, _ = read_tiled_crop(
                tif_path, x0, y0, args.block_px, args.block_px
            )

        stem = f"k{k:03d}"
        uav_out = set_dir / f"{stem}_uav.jpg"
        block_out = set_dir / f"{stem}_block.jpg"
        uav_256.save(uav_out, quality=95)
        Image.fromarray(block, mode="RGB").save(block_out, quality=95)

        phi1 = float(row["Phi1"])
        theta = (args.theta_sign * phi1 + args.theta_offset) % 360.0

        records.append(
            {
                "set": set_name,
                "k": k,
                "row": source_row_index,
                "x_norm": 0.0,
                "y_norm": 0.0,
                "theta": theta,
                "source_filename": filename,
                "lat": row["lat"],
                "lon": row["lon"],
                "height": row["height"],
                "Phi1": row["Phi1"],
                "Phi2": row["Phi2"],
                "map_center_x": cx,
                "map_center_y": cy,
                "uav_crop_px": args.uav_crop_px,
            }
        )
        print(
            f"[{k:02d}/{args.n}] {filename} -> {uav_out.name}, {block_out.name} "
            f"(map centre {cx},{cy}; theta={theta:.3f})"
        )

    # Trone-iMatch required columns first; additional provenance columns follow.
    csv_out = out_root / "samples.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(csv_out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    note = out_root / "UAV_VISLOC_PROVISIONAL_README.txt"
    note.write_text(
        f"""PROVISIONAL UAV-VisLoc export
================================
Set: {set_name}
Samples: {args.n}
UAV input: central {args.uav_crop_px}x{args.uav_crop_px} crop resized to 256x256
Satellite input: GPS-centred {args.block_px}x{args.block_px} crop
Ground-truth centre: x_norm=0, y_norm=0 (GPS-centred block)
Heading: theta = ({args.theta_sign}) * Phi1 + ({args.theta_offset}) degrees, modulo 360

IMPORTANT LIMITATIONS
---------------------
1. The example README does not provide camera intrinsics / calibrated field of view.
2. The UAV crop size ({args.uav_crop_px}px) is a scale assumption, not a measured footprint.
3. Phi1 is used provisionally; image-axis / map-axis sign and offset must be verified.
4. The GPS coordinate accuracy and image-to-map timestamp alignment have not been
   independently checked.
5. These samples are useful for pipeline smoke-testing, but results must not be
   presented as a validated quantitative external benchmark until these assumptions
   are verified.

Run the exporter with --help to inspect configurable parameters.
""",
        encoding="utf-8",
    )

    print("\nEXPORT COMPLETE (PROVISIONAL)")
    print("samples.csv:", csv_out.resolve())
    print("image folder:", set_dir.resolve())
    print("caveats:", note.resolve())


if __name__ == "__main__":
    main()
