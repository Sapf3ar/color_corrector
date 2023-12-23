import numpy as np
from typing import DefaultDict
import cv2

def contrast_score(
    img: np.ndarray, 
    blur_kernel: tuple[int, int] = (11, 11)
    ) -> np.float64:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    contrast = blurred.std()
    return contrast

def luminance_score(img: np.ndarray) -> np.float64:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    luminance = img.reshape(-1, 3)[:, 1].mean()
    return luminance


def features(image: np.ndarray) -> DefaultDict:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    grayness = 1 - cv2.mean(gray)[0] / 255.0
    
    saturation = (hsv[:, :, 1] / 255).mean()
    saturation_std = (hsv[:, :, 1] / 255).std()
    
    brightness = (hsv[:, :, 2] / 255).mean()
    brightness_std = (hsv[:, :, 2] / 255).std()
    
    blure = cv2.Laplacian(rgb, cv2.CV_64F).var()
    
    contrast = contrast_score(rgb)
    
    luminance = luminance_score(rgb)
    
    red = (rgb[:, :, 0] / 255).mean()
    red_std = (rgb[:, :, 0] / 255).std()
    red_count = ((rgb[:, :, 0] / 255) > 0.5).sum() / (rgb.shape[0] * rgb.shape[1])
    
    green = (rgb[:, :, 1] / 255).mean()
    green_std = (rgb[:, :, 1] / 255).std()
    green_count = ((rgb[:, :, 1] / 255) > 0.5).sum() / (rgb.shape[0] * rgb.shape[1])

    blue = (rgb[:, :, 2] / 255).mean()
    blue_std = (rgb[:, :, 2] / 255).std()
    blue_count = ((rgb[:, :, 2] / 255) > 0.5).sum() / (rgb.shape[0] * rgb.shape[1])
    
    image_features = {
                    "saturation":saturation,
                    "brightness":brightness,
                    "contrast":contrast,
                    "red":red,
                    "gren":green,
                    "blue":blue
                    }
    return image_features




