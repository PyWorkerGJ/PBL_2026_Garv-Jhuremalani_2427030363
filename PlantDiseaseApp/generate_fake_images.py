import cv2
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

input_folder = "training_data/real"
output_folder = "training_data/fake"
os.makedirs(output_folder, exist_ok=True)


def too_smooth(img):
    """AI images are often over-smoothed with perfect, waxy textures."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    smoothed = pil.filter(ImageFilter.SMOOTH_MORE)
    smoothed = pil.filter(ImageFilter.SMOOTH_MORE)  
    smoothed = ImageEnhance.Sharpness(smoothed).enhance(0.3)  
    return cv2.cvtColor(np.array(smoothed), cv2.COLOR_RGB2BGR)

def too_perfect_colors(img):
    """AI images have overly saturated, 'unnatural' perfect colors."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Color(pil).enhance(np.random.uniform(1.8, 2.5))      
    pil = ImageEnhance.Brightness(pil).enhance(np.random.uniform(1.1, 1.3)) 
    pil = ImageEnhance.Contrast(pil).enhance(np.random.uniform(1.2, 1.6))  
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def frequency_artifacts(img):
    """
    AI images have unusual high-frequency patterns in the noise floor.
    We simulate this by adding structured periodic noise.
    """
    h, w = img.shape[:2]
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    xv, yv = np.meshgrid(x, y)
    pattern = (np.sin(xv) * np.cos(yv) * 8).astype(np.int16)
    result = img.astype(np.int16) + pattern[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)

def unnatural_background(img):
    """AI images often have unnaturally uniform or gradient backgrounds."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w//2, h//2), (w//3, h//2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    bg_color = np.array([
        np.random.randint(180, 240),
        np.random.randint(180, 240),
        np.random.randint(180, 240)
    ], dtype=np.float32)
    bg = np.ones((h, w, 3), dtype=np.float32) * bg_color
    mask3 = mask[:, :, np.newaxis]
    blended = img.astype(np.float32) * mask3 + bg * (1 - mask3)
    return blended.astype(np.uint8)

def jpeg_dreamlike(img):
    """
    AI images often have a dreamlike softness with slight halo around edges.
    """
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    blurred = pil.filter(ImageFilter.GaussianBlur(radius=4))
    blended = Image.blend(pil, blurred, alpha=0.35)
    blended = ImageEnhance.Sharpness(blended).enhance(0.5)
    return cv2.cvtColor(np.array(blended), cv2.COLOR_RGB2BGR)

def texture_repetition(img):
    """
    AI images sometimes have subtly repeating texture patterns.
    """
    h, w = img.shape[:2]
    tile_h, tile_w = h // 3, w // 3
    tile = img[:tile_h, :tile_w].copy()
    result = img.copy()
    x1 = np.random.randint(0, w - tile_w)
    y1 = np.random.randint(0, h - tile_h)
    region = result[y1:y1+tile_h, x1:x1+tile_w].astype(np.float32)
    blended = (region * 0.7 + tile.astype(np.float32) * 0.3).astype(np.uint8)
    result[y1:y1+tile_h, x1:x1+tile_w] = blended
    return result

def subtle_noise(img):
    """Very subtle fine-grain noise — much less extreme than old version."""
    noise = np.random.normal(0, 4, img.shape).astype(np.float32)  
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)

def color_channel_bias(img):
    """AI images sometimes have a slight color cast in one channel."""
    result = img.copy().astype(np.float32)
    channel = np.random.randint(0, 3)
    bias = np.random.uniform(1.05, 1.20)
    result[:, :, channel] *= bias
    return np.clip(result, 0, 255).astype(np.uint8)


effects = [
    too_smooth,
    too_perfect_colors,
    frequency_artifacts,
    unnatural_background,
    jpeg_dreamlike,
    texture_repetition,
    subtle_noise,
    color_channel_bias,
]

combo_effects = [
    (too_smooth, too_perfect_colors),
    (jpeg_dreamlike, too_perfect_colors),
    (frequency_artifacts, subtle_noise),
    (unnatural_background, too_smooth),
    (color_channel_bias, jpeg_dreamlike),
]


count = 0
skipped = 0

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    if img is None:
        skipped += 1
        continue

    for eff in effects:
        try:
            fake = eff(img.copy())
            out_path = os.path.join(output_folder, f"fake_{count}.jpg")
            cv2.imwrite(out_path, fake)
            count += 1
        except Exception as e:
            print(f"Skipped {eff.__name__} on {filename}: {e}")

    for eff1, eff2 in combo_effects:
        try:
            fake = eff2(eff1(img.copy()))
            out_path = os.path.join(output_folder, f"fake_{count}.jpg")
            cv2.imwrite(out_path, fake)
            count += 1
        except Exception as e:
            print(f"Skipped combo on {filename}: {e}")

print(f"\nGenerated {count} fake images from {len(os.listdir(input_folder)) - skipped} real images.")
print(f"Skipped {skipped} unreadable files.")
print(f"Fake images saved to: {output_folder}")
print(f"\nNext step: run python retrain.py")
