import cv2
import numpy as np

def preprocess_image(image_path, img_height=32, img_width=128):
    """
    Preprocess the input image:
    - Read image in grayscale
    - Resize to fixed height and width (padding if needed)
    - Normalize pixel values to [0,1]
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h, w = img.shape
    new_w = int(w * (img_height / h))
    img = cv2.resize(img, (new_w, img_height))
    if new_w > img_width:
        img = cv2.resize(img, (img_width, img_height))
        new_w = img_width
    # Padding to img_width
    padded_img = np.ones((img_height, img_width), dtype=np.uint8) * 255
    padded_img[:, :new_w] = img
    # Normalize to [0,1]
    normalized_img = padded_img.astype(np.float32) / 255.0
    return normalized_img
