import cv2
import os
import numpy as np
from PIL import Image

input_folder = "training_data/real"
output_folder = "training_data/fake"
os.makedirs(output_folder, exist_ok=True)

def random_noise(img):
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    return cv2.add(img, noise)

def random_color_shift(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,1] = cv2.add(hsv[:,:,1], np.random.randint(-40, 40))
    hsv[:,:,2] = cv2.add(hsv[:,:,2], np.random.randint(-50, 50))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def random_blur(img):
    k = np.random.choice([3,5,7])
    return cv2.GaussianBlur(img, (k,k), 0)

def random_patch(img):
    h, w, _ = img.shape
    x1 = np.random.randint(0, w//2)
    y1 = np.random.randint(0, h//2)
    x2 = x1 + np.random.randint(30, 80)
    y2 = y1 + np.random.randint(30, 80)
    img[y1:y2, x1:x2] = np.random.randint(0,255,(y2-y1, x2-x1,3))
    return img

effects = [random_noise, random_color_shift, random_blur, random_patch]

count = 0
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    for eff in effects:
        fake = eff(img.copy())
        out_path = os.path.join(output_folder, f"fake_{count}.jpg")
        cv2.imwrite(out_path, fake)
        count += 1
print(f"Generated {count} fake images!")
