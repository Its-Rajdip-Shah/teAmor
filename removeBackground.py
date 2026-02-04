from PIL import Image
import numpy as np
import os
from collections import deque

# Chroma key color (background)
key_color = np.array([46, 230, 17])  # #2EE611
# Additional green key color
key_color_alt = np.array([0, 255, 130])  # #00FF82
# Gemini green background key (shades)
green_key_0df50b = np.array([13, 245, 11])  # #0DF50B
# White background key (optional)
white_key = np.array([248, 252, 235])  # #F8FCEB
# Gray background keys (shades)
gray_keys = [
    np.array([157, 159, 157]),  # #9D9F9D
    np.array([190, 190, 190]),  # #BEBEBE
]
# Yellow background key (shades)
yellow_key = np.array([254, 226, 2])  # #FEE202
# Magenta background key (shades)
magenta_key = np.array([255, 0, 255])  # #FF00FF

# Tuning knobs (increase dist for more aggressive removal)
dist_threshold = 200
min_green = 90
green_dominance = 10  # g must exceed max(r, b) by this amount
white_dist_threshold = 80
white_min_channel = 210
remove_white = False
gray_dist_threshold = 35
gray_max_chroma = 18  # max(channel) - min(channel) for grayish pixels
remove_gray = False
yellow_dist_threshold = 140
yellow_min_r = 200
yellow_min_g = 180
yellow_max_b = 90
remove_yellow = False
magenta_dist_threshold = 120
magenta_min_r = 200
magenta_min_b = 200
magenta_max_g = 60
remove_magenta = True
green_0df50b_dist_threshold = 140
remove_green_0df50b = False
remove_green_screen = False
despill_magenta = True
despill_threshold = 15
despill_strength = 0.8  # 0..1, higher = stronger despill
edge_ring_radius = 1
remove_magenta_edge_ring = True

def remove_bg(input_path: str, output_path: str) -> str:

    # Load image
    img = Image.open(input_path).convert("RGBA")
    data = np.array(img)

    # Split channels (use signed ints to avoid underflow)
    r = data[..., 0].astype(np.int16)
    g = data[..., 1].astype(np.int16)
    b = data[..., 2].astype(np.int16)

    # Distance to green key colors
    dr = r - int(key_color[0])
    dg = g - int(key_color[1])
    db = b - int(key_color[2])
    dist = np.sqrt(dr * dr + dg * dg + db * db)

    dr2 = r - int(key_color_alt[0])
    dg2 = g - int(key_color_alt[1])
    db2 = b - int(key_color_alt[2])
    dist2 = np.sqrt(dr2 * dr2 + dg2 * dg2 + db2 * db2)

    mask = np.zeros_like(r, dtype=bool)

    # Only remove pixels that are clearly green-screen (edge-connected to protect foreground greens)
    if remove_green_screen:
        green_dom = g - np.maximum(r, b)
        green_candidate = ((dist <= dist_threshold) | (dist2 <= dist_threshold)) & (g >= min_green) & (green_dom >= green_dominance)
        mask |= edge_connected_mask(green_candidate)

    # Also remove near-white background (optional)
    if remove_white:
        wr = r - int(white_key[0])
        wg = g - int(white_key[1])
        wb = b - int(white_key[2])
        white_dist = np.sqrt(wr * wr + wg * wg + wb * wb)
        white_mask = (
            (white_dist <= white_dist_threshold) &
            (r >= white_min_channel) &
            (g >= white_min_channel) &
            (b >= white_min_channel)
        )
        mask |= white_mask

    # Remove gray backgrounds around #9D9F9D / #BEBEBE (and nearby shades)
    if remove_gray:
        chroma = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])
        grayish = chroma <= gray_max_chroma
        gray_mask = np.zeros_like(mask, dtype=bool)
        for gkey in gray_keys:
            gr = r - int(gkey[0])
            gg = g - int(gkey[1])
            gb = b - int(gkey[2])
            gdist = np.sqrt(gr * gr + gg * gg + gb * gb)
            gray_mask |= (gdist <= gray_dist_threshold) & grayish
        mask |= gray_mask

    # Remove yellow backgrounds around #FEE202 (and nearby shades)
    if remove_yellow:
        yr = r - int(yellow_key[0])
        yg = g - int(yellow_key[1])
        yb = b - int(yellow_key[2])
        ydist = np.sqrt(yr * yr + yg * yg + yb * yb)
        yellow_mask = (
            (ydist <= yellow_dist_threshold) &
            (r >= yellow_min_r) &
            (g >= yellow_min_g) &
            (b <= yellow_max_b)
        )
        mask |= yellow_mask

    # Remove magenta backgrounds around #FF00FF (and nearby shades)
    if remove_magenta:
        mr = r - int(magenta_key[0])
        mg = g - int(magenta_key[1])
        mb = b - int(magenta_key[2])
        mdist = np.sqrt(mr * mr + mg * mg + mb * mb)
        magenta_candidate = (
            (mdist <= magenta_dist_threshold) &
            (r >= magenta_min_r) &
            (b >= magenta_min_b) &
            (g <= magenta_max_g)
        )
        mask |= edge_connected_mask(magenta_candidate)
        if remove_magenta_edge_ring:
            edge_ring = dilate_mask(mask, edge_ring_radius) & ~mask
            mask |= edge_ring & magenta_candidate

    # Remove aggressive green background around #0DF50B, but only where it connects to edges
    if remove_green_0df50b:
        gr = r - int(green_key_0df50b[0])
        gg = g - int(green_key_0df50b[1])
        gb = b - int(green_key_0df50b[2])
        gdist = np.sqrt(gr * gr + gg * gg + gb * gb)
        candidate = gdist <= green_0df50b_dist_threshold
        edge_mask = edge_connected_mask(candidate)
        mask |= edge_mask

    # Despill magenta fringing on edge pixels that remain
    if despill_magenta and remove_magenta:
        edge_ring = dilate_mask(mask, edge_ring_radius) & ~mask
        apply_magenta_despill(data, edge_ring)

    # Apply transparency
    data[..., 3][mask] = 0

    Image.fromarray(data).save(output_path)
    return output_path


def edge_connected_mask(candidate: np.ndarray) -> np.ndarray:
    h, w = candidate.shape
    visited = np.zeros_like(candidate, dtype=bool)
    q = deque()

    # Seed with edge pixels that match candidate
    for x in range(w):
        if candidate[0, x]:
            q.append((0, x))
            visited[0, x] = True
        if candidate[h - 1, x]:
            q.append((h - 1, x))
            visited[h - 1, x] = True
    for y in range(h):
        if candidate[y, 0] and not visited[y, 0]:
            q.append((y, 0))
            visited[y, 0] = True
        if candidate[y, w - 1] and not visited[y, w - 1]:
            q.append((y, w - 1))
            visited[y, w - 1] = True

    # Flood fill through candidate pixels
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and candidate[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    return visited


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def apply_magenta_despill(data: np.ndarray, edge_mask: np.ndarray) -> None:
    if not np.any(edge_mask):
        return
    r = data[..., 0].astype(np.float32)
    g = data[..., 1].astype(np.float32)
    b = data[..., 2].astype(np.float32)
    cast = (r + b) * 0.5 - g
    spill = edge_mask & (cast > despill_threshold)
    if not np.any(spill):
        return
    adjust = cast * despill_strength
    r[spill] = np.maximum(0, r[spill] - adjust[spill])
    b[spill] = np.maximum(0, b[spill] - adjust[spill])
    g[spill] = np.minimum(255, g[spill] + adjust[spill] * 0.25)
    data[..., 0] = np.clip(r, 0, 255).astype(np.uint8)
    data[..., 1] = np.clip(g, 0, 255).astype(np.uint8)
    data[..., 2] = np.clip(b, 0, 255).astype(np.uint8)


def process_dir(input_dir: str) -> str:
    output_dir = os.path.join(input_dir, "transparent")
    os.makedirs(output_dir, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    count = 0
    for name in os.listdir(input_dir):
        src = os.path.join(input_dir, name)
        if not os.path.isfile(src):
            continue
        if os.path.splitext(name)[1].lower() not in exts:
            continue
        dst = os.path.join(output_dir, name)
        remove_bg(src, dst)
        count += 1
    return f"{output_dir} ({count} files)"


while True:
    user_input = input("Image path (blank/exit to quit): ").strip()
    if not user_input or user_input.lower() in {"exit", "quit"}:
        break
    # Strip surrounding quotes and expand ~
    cleaned = user_input.strip().strip("\"'").strip()
    cleaned = os.path.expanduser(cleaned)
    cleaned = os.path.normpath(cleaned)
    try:
        if os.path.isdir(cleaned):
            out = process_dir(cleaned)
            print(f"✅ Saved: {out}")
        else:
            out = remove_bg(cleaned, "output_transparent.png")
            print(f"✅ Saved: {out}")
    except Exception as exc:
        print(f"❌ Error: {exc}")
