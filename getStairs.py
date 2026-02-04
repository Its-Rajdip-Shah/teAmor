from PIL import Image
import numpy as np
import json

# Paths
input_paths = [
    ("environment/left top to right bottom.png", "environment/left top to right bottom.json"),
    ("environment/left bottom to right top.png", "environment/left bottom to right top.json"),
]

# Target color (#00FF82) and tolerance per channel
target = np.array([0, 255, 130], dtype=np.int16)
tol = 6  # lower = stricter

for input_path, output_path in input_paths:
    img = Image.open(input_path).convert("RGB")
    data = np.array(img, dtype=np.int16)

    # Mask for pixels close to #00FF82
    r = data[..., 0]
    g = data[..., 1]
    b = data[..., 2]
    mask = (
        (np.abs(r - target[0]) <= tol) &
        (np.abs(g - target[1]) <= tol) &
        (np.abs(b - target[2]) <= tol)
    )

    # Extract coordinates (x, y)
    ys, xs = np.where(mask)
    coords = list(zip(xs.tolist(), ys.tolist()))

    # Save
    payload = {
        "width": img.width,
        "height": img.height,
        "coords": coords
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"✅ Saved {len(coords)} stair pixels to {output_path}")
