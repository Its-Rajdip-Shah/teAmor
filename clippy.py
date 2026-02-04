from PIL import Image
import numpy as np

# Paths
front_path = "environment/front.png"
back_path = "environment/back.png"
output_path = "environment/masked_output.png"

# Target paint color (#00FF82) and tight per-channel tolerance
target_color = np.array([0, 255, 130], dtype=np.int16)
tol_r = 12
tol_g = 12
tol_b = 12

# Load images
front = Image.open(front_path).convert("RGBA")
back = Image.open(back_path).convert("RGBA")

if front.size != back.size:
    raise ValueError("front.png and back.png must be the same resolution.")

front_data = np.array(front, dtype=np.int16)
back_data = np.array(back, dtype=np.int16)

# Compute per-channel distance from target color on RGB channels
fr = front_data[..., 0]
fg = front_data[..., 1]
fb = front_data[..., 2]

mask = (
    (np.abs(fr - target_color[0]) <= tol_r) &
    (np.abs(fg - target_color[1]) <= tol_g) &
    (np.abs(fb - target_color[2]) <= tol_b)
)

# Build output: keep back pixels only where mask is true
out = np.zeros_like(back_data, dtype=np.uint8)
out[..., :3] = back_data[..., :3].astype(np.uint8)
out[..., 3] = (mask * 255).astype(np.uint8)

# Save
Image.fromarray(out, mode="RGBA").save(output_path)
print(f"✅ Saved: {output_path}")
