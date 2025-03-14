import random
import numpy as np
import cv2
from datasets import load_dataset


def random_class(class_names=None, n_class=20, rand_seed=42):
    """
    randomly pick the classes
    class_names = list
    n = number of class
    rand_seed = random seed
    """
    if class_names is None:
        raise ValueError(f"Class names cannot be None")

    random.seed(rand_seed)
    selected_classes = random.sample(class_names, n_class)
    print("Selected classes:", selected_classes)
    return selected_classes


def process_image(image, image_size=(224, 224)):
    """
    resizes and
    normalizes values to [0, 1].
    """
    # numpy error
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        
    # resizing
    image = cv2.resize(image, image_size)
    
    # Normalising
    image = image.astype("float32") / 255.0
    return image


def load_food(image_size=(224, 224), rand_seed=42, n_class=20 ):
    """
    input:
    image_size using tuple default (224,224)
    rand_seed default 42
    n_class to take out default 20
    returns training and val dataset of food
    dataset.features["label"].names will contain the label names, using the label numbers
    dataset[n] contains {'image': image as np array, 'label': label number}
    """
    
    train_ds = load_dataset("ethz/food101", split="train")
    val_ds = load_dataset("ethz/food101", split="validation")
    # print("Available splits in Food101 dataset:", list(ds.keys()))
    
    # get all names and pick some (n_class)
    class_names = train_ds.features["label"].names
    selected_classes = random_class(class_names, n_class, rand_seed)
    
    # get the label number
    indices = [class_names.index(c) for c in selected_classes]
    
    # filter them out
    train_ds = train_ds.filter(lambda x: x['label'] in indices)
    val_ds = val_ds.filter(lambda x: x['label'] in indices)
    
    # resizing and normalizing
    train_ds = train_ds.map(lambda x: {"image": process_image(x["image"], image_size)}, num_proc=4)
    val_ds = val_ds.map(lambda x: {"image": process_image(x["image"], image_size)}, num_proc=4)
    
    train_ds.set_format("numpy", columns=["image"])
    val_ds.set_format("numpy", columns=["image"])
    
    return train_ds, val_ds
