from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
from PIL import Image


DEFAULT_INPUT = "environment/breaking.png"
DEFAULT_OUTPUT = "environment/breaking.png"
DEFAULT_KEY = (217, 217, 191)  # #D9D9BF
DEFAULT_TOLERANCE = 20


def parse_hex_color(value: str) -> Tuple[int, int, int]:
    v = value.strip().lstrip("#")
    if len(v) != 6:
        raise ValueError("Color must be in RRGGBB format.")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)


def remove_color_key(img: Image.Image, key_rgb: Tuple[int, int, int], tol: int) -> Image.Image:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove a key color from breaking.png and overwrite output.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input image path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output image path (overwrite).")
    parser.add_argument("--key", default="D9D9BF", help="Key color in hex (RRGGBB).")
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE, help="Color distance tolerance.")
    args = parser.parse_args()

    key_rgb = parse_hex_color(args.key)
    img = Image.open(args.input)
    out = remove_color_key(img, key_rgb, args.tolerance)
    out.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
