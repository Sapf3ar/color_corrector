import numpy as np
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


def features(image: np.ndarray) -> tuple:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold = (cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4) == 0)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)    
    
    lightness = (hls[:, :, 1] / 255).mean()
    lightness_std = (hls[:, :, 1] / 255).std()
    
    grayness = 1 - cv2.mean(gray)[0] / 255.0
    
    saturation = (hsv[:, :, 1] / 255).mean()
    saturation_std = (hsv[:, :, 1] / 255).std()
    
    brightness = (hsv[:, :, 2] / 255).mean()
    brightness_std = (hsv[:, :, 2] / 255).std()
    
    blure = cv2.Laplacian(rgb, cv2.CV_64F).var()
    
    contrast = contrast_score(rgb)
    
    luminance = luminance_score(rgb)
    
    red_im = np.ma.array(rgb[:, :, 0], mask=threshold)
    red = (rgb[:, :, 0] / 255).mean()
    red_th = (red_im / 255).mean()
    red_std = (red_im / 255).std()
    red_count = ((red_im / 255) > 0.7).sum() / red_im.sum()
    
    green_im = np.ma.array(rgb[:, :, 1], mask=threshold)
    green = (rgb[:, :, 1] / 255).mean()
    green_th = (green_im / 255).mean()
    green_std = (green_im / 255).std()
    green_count = ((green_im / 255) > 0.7).sum() / green_im.sum()

    blue_im = np.ma.array(rgb[:, :, 2], mask=threshold)
    blue = (rgb[:, :, 2] / 255).mean()
    blue_th = (blue_im / 255).mean()
    blue_std = (blue_im / 255).std()
    blue_count = ((blue_im / 255) > 0.7).sum() / blue_im.sum()
    
    return grayness, saturation, saturation_std, brightness, brightness_std, \
           lightness, lightness_std, \
           blure, contrast, luminance, red, red_th, red_std, red_count, \
           green, green_th, green_std, green_count, blue, blue_th, blue_std, blue_count
