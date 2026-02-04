from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
from PIL import Image


DEFAULT_INPUT = "cracked shit.png"
DEFAULT_OUT_DIR = "cracked_shit_frames"
DEFAULT_COLS = 3
DEFAULT_ROWS = 2
DEFAULT_FRAME_W = 1045
DEFAULT_FRAME_H = 694
DEFAULT_KEY = (218, 216, 222)  # #DAD8DE
DEFAULT_TOLERANCE = 18


def remove_bg_rgba(img: Image.Image, key_rgb: Tuple[int, int, int], tol: int) -> Image.Image:
    data = np.array(img.convert("RGBA"))
    r = data[..., 0].astype(np.int16)
    g = data[..., 1].astype(np.int16)
    b = data[..., 2].astype(np.int16)
    dr = r - int(key_rgb[0])
    dg = g - int(key_rgb[1])
    db = b - int(key_rgb[2])
    dist = np.sqrt(dr * dr + dg * dg + db * db)
    mask = dist <= tol
    data[..., 3][mask] = 0
    return Image.fromarray(data)


def split_sheet(
    input_path: str,
    out_dir: str,
    cols: int,
    rows: int,
    frame_w: int,
    frame_h: int,
    key_rgb: Tuple[int, int, int],
    tol: int,
) -> int:
    img = Image.open(input_path).convert("RGBA")
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for row in range(rows):
        for col in range(cols):
            left = col * frame_w
            top = row * frame_h
            box = (left, top, left + frame_w, top + frame_h)
            frame = img.crop(box)
            frame = remove_bg_rgba(frame, key_rgb, tol)
            out_name = os.path.join(out_dir, f"frame_{count:02d}.png")
            frame.save(out_name)
            count += 1
    return count


def parse_hex_color(value: str) -> Tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) != 6:
        raise ValueError("Color must be in RRGGBB format.")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a sprite sheet into frames and remove a key color.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input sprite sheet path.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for frames.")
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS, help="Number of columns.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Number of rows.")
    parser.add_argument("--frame-w", type=int, default=DEFAULT_FRAME_W, help="Frame width in pixels.")
    parser.add_argument("--frame-h", type=int, default=DEFAULT_FRAME_H, help="Frame height in pixels.")
    parser.add_argument("--key", default="DAD8DE", help="Background key color in hex (RRGGBB).")
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE, help="Color distance tolerance.")
    args = parser.parse_args()

    key_rgb = parse_hex_color(args.key)
    count = split_sheet(
        args.input,
        args.out_dir,
        args.cols,
        args.rows,
        args.frame_w,
        args.frame_h,
        key_rgb,
        args.tolerance,
    )
    print(f"Saved {count} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
