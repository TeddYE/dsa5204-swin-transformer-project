from os import path

CACHE_DIR = path.join(path.abspath(path.dirname(__file__)), 'data', 'cache')
OUTPUT_DIR = path.join(path.abspath(path.dirname(__file__)), 'output')
IMAGE_DIR = path.join(path.abspath(path.dirname(__file__)), 'data', 'images')
GRAD_CAM_DIR = path.join(path.abspath(path.dirname(__file__)), 'data', 'gradcam')
