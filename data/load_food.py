import random

import numpy as np
import cv2

from datasets import load_dataset
from config import CACHE_DIR
from utils import logger

logger = logger.get_logger(__name__)


def random_class(class_names=None, n_class=20, rand_seed=42):
    """
    randomly pick the classes
    class_names = list
    n = number of class
    rand_seed = random seed
    """

    random.seed(rand_seed)
    selected_classes = random.sample(class_names, n_class)
    print("Selected classes:", selected_classes)
    return selected_classes


def process_image(image, image_size=(224, 224)):
    # numpy error
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Resize so shorter side = 256 (maintain aspect ratio)
    h, w = image.shape[:2]
    if h < w:
        new_h, new_w = 256, int(w * 256 / h)
    else:
        new_w, new_h = 256, int(h * 256 / w)
    image = cv2.resize(image, (new_w, new_h))

    # Center crop to image_size
    crop_w, crop_h = image_size
    start_x = (new_w - crop_w) // 2
    start_y = (new_h - crop_h) // 2
    image = image[start_y:start_y + crop_h, start_x:start_x + crop_w]

    # scaling
    image = image.astype("float32") / 255.0

    # normalizing using ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    return image


def load_food(image_size=(224, 224), rand_seed=42, n_class=20):

    logger.info("Loading food data...")

    train_ds = load_dataset("ethz/food101", split="train", cache_dir=CACHE_DIR)
    val_ds = load_dataset("ethz/food101", split="validation", cache_dir=CACHE_DIR)

    logger.info(f"Loaded train_ds {len(train_ds)} images and val_ds {len(val_ds)} images...")

    # get all names and pick some (n_class)
    class_names = train_ds.features["label"].names
    selected_classes = random_class(class_names, n_class, rand_seed)

    # get the label number
    indices = [class_names.index(c) for c in selected_classes]

    # filter them out
    train_ds = train_ds.filter(lambda x: x['label'] in indices)
    val_ds = val_ds.filter(lambda x: x['label'] in indices)

    logger.info(f"Filtered train_ds {len(train_ds)} images and val_ds {len(val_ds)} images...")

    # resizing and normalizing
    train_ds = train_ds.map(lambda x: {"image": process_image(x["image"], image_size)}, num_proc=4)
    val_ds = val_ds.map(lambda x: {"image": process_image(x["image"], image_size)}, num_proc=4)

    train_ds.set_format("numpy", columns=["image"])
    val_ds.set_format("numpy", columns=["image"])

    logger.info("Processed train_ds and val_ds...")

    return train_ds, val_ds
