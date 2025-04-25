from transformers import VisionEncoderDecoderModel, GPT2TokenizerFast, ViTImageProcessor
import torch
import os
import urllib.parse as parse
from PIL import Image
import requests
from config import IMAGE_DIR

# a function to determine whether a string is a URL or not
def is_url(string):
    try:
        result = parse.urlparse(string)
        return all([result.scheme, result.netloc, result.path])
    except:
        return False


# a function to load an image
def load_image(image_path):
    if is_url(image_path):
        return Image.open(requests.get(image_path, stream=True).raw)
    elif os.path.exists(image_path):
        return Image.open(image_path)


# a function to perform inference
def get_caption(model, image_processor, tokenizer, image_path):
    image = load_image(image_path)
    # preprocess the image
    img = image_processor(image, return_tensors="pt").to(device)
    # generate the caption (using greedy decoding by default)
    output = model.generate(**img)
    # decode the output
    caption = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return caption


device = "cuda" if torch.cuda.is_available() else "cpu"
# load the fine-tuned image captioning model and corresponding tokenizer and image processor
model = VisionEncoderDecoderModel.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("Abdou/vit-swin-base-224-gpt2-image-captioning")

image_paths = [os.path.join(IMAGE_DIR, fname) for fname in sorted(os.listdir(IMAGE_DIR)) if fname.endswith(".jpg") or fname.endswith(".jpeg") or fname.endswith(".png")]

image_path = image_paths[0]

# get the caption
caption = get_caption(model, image_processor, tokenizer, image_path)
print(f"caption: {caption}")
