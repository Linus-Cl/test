import cv2
import os

HERO_TEMPLATES_PATH = "hero_templates/"

for filename in os.listdir(HERO_TEMPLATES_PATH):
    if filename.endswith(".png"):
        filepath = os.path.join(HERO_TEMPLATES_PATH, filename)
        img = cv2.imread(filepath)
        if img is not None:
            resized_img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)
            cv2.imwrite(filepath, resized_img)
            print(f"Resized {filename} to 80x80")