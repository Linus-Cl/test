import cv2
import os

HERO_TEMPLATES_PATH = "hero_templates/"

hero_templates = {
    os.path.basename(p).split(".")[0]: cv2.imread(
        os.path.join(HERO_TEMPLATES_PATH, p)
    )
    for p in os.listdir(HERO_TEMPLATES_PATH)
    if p.endswith(".png")
}

print("--- Hero Template Dimensions (Width x Height) ---")
for name, template in hero_templates.items():
    if template is not None:
        h, w, _ = template.shape
        print(f"{name:<15}: {w} x {h}")
    else:
        print(f"{name:<15}: FAILED TO LOAD")
