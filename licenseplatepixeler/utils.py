import cv2
import json
import os

CONFIG_PATH = "config.json"

def add_pixelation(image, boxes, pixel_size=10):
    for (x1, y1, x2, y2) in boxes:
        region = image[y1:y2, x1:x2]
        region = cv2.resize(region, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
        region = cv2.resize(region, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
        image[y1:y2, x1:x2] = region
    return image

def setup_logging(verbose):
    import logging
    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING)

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {}

def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
