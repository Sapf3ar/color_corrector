# import the necessary packages
import numpy as np
import cv2
import streamlit as st
from typing import Tuple
from PIL import Image
from numpy.typing import NDArray
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os
st.session_state.show_button=False
import cv2
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

st.session_state.export_data = None
@st.cache_data
def get_depth_model():

    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    return model
class Midas():
    def __init__(
        self,
    ):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = get_depth_model()
        self.model = self.model.to(self.device)
        self.model = self.model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def get_depth_map(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(image).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()
        return output

scaler = MinMaxScaler()

def depth_transform(source, max_depth_th=0.7, min_depth_th=0.2, depth_model=None):
    source_depth = depth_model.get_depth_map(source)
    
    source_depth = scaler.fit_transform(source_depth)
    middleground_source = (source_depth >= max_depth_th) & (source_depth <= min_depth_th)
    source_mask = np.where(middleground_source, source[:, :, 0], 0)
    source_mask1 = np.where(middleground_source, source[:, :, 1], 0)
    source_mask2 = np.where(middleground_source, source[:, :, 2], 0)
    source = np.stack([source_mask, source_mask1, source_mask2], axis=-1 )
    return source

def full_color_transfer(source, target,clip=True, preserve_paper=True):
    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l = (lStdSrc / lStdTar) * l
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)
    return (l, a, b)
def stringify_array(x):
    return [str(i) for i in x]

def get_threshold_filter(source, source_gray):
    mask = cv2.adaptiveThreshold(source_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    
    source_mask = np.where(mask, source[:, :, 0], 0.0)
    source_mask1 = np.where(mask, source[:, :, 1], 0.0)
    source_mask2 = np.where(mask, source[:, :, 2], 0.0)
    source = np.stack([source_mask, source_mask1, source_mask2], axis=-1 )
    source = source.astype(np.float32)

    return source

def color_transfer(source, target, clip=True, preserve_paper=True, depth_model=None):
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.

    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.

    Parameters:
    -------
    source: NumPy array
        OpenCV image in BGR color space (the source image)
    target: NumPy array
        OpenCV image in BGR color space (the target image)
    clip: Should components of L*a*b* image be scaled by np.clip before 
        converting back to BGR color space?
        If False then components will be min-max scaled appropriately.
        Clipping will keep target image brightness truer to the input.
        Scaling will adjust image brightness to avoid washed out portions
        in the resulting color transfer that can be caused by clipping.
    preserve_paper: Should color transfer strictly follow methodology
        layed out in original paper? The method does not always produce
        aesthetically pleasing results.
        If False then L*a*b* components will scaled using the reciprocal of
        the scaling factor proposed in the paper.  This method seems to produce
        more consistently aesthetically pleasing results 

    Returns:
    -------
    transfer: NumPy array
        OpenCV image (w, h, 3) NumPy array (uint8)
    """
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)


    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    
    source_thresholded = get_threshold_filter(source=source, source_gray=source_gray)
    l2, a2, b2 = full_color_transfer(source_thresholded, target)

    l, a, b = full_color_transfer(source, target)
    
    source_depth = depth_transform(source, depth_model=depth_model)
    l1, a1, b1 = full_color_transfer(source_depth, target)

    export_params = {
        "full_source":stringify_array(image_stats(source)),
        "depth_transformed_source":stringify_array(image_stats(source_depth)),
        "light_transformed_source":stringify_array(image_stats(source_thresholded))
    }    
    st.session_state.export_data = export_params

    l = np.stack([l, l1, l2], axis=-1).mean(-1)
    a = np.stack([a, a1, a2], axis=-1).mean(-1)
    b = np.stack([b, b1, b2], axis=-1).mean(-1)
    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
    
    # return the color transferred image
    return transfer

def image_stats(image):
    """
    Parameters:
    -------
    image: NumPy array
        OpenCV image in L*a*b* color space

    Returns:
    -------
    Tuple of mean and standard deviations for the L*, a*, and b*
    channels, respectively
    """
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)

def _min_max_scale(arr, new_range=(0, 255)):
    """
    Perform min-max scaling to a NumPy array

    Parameters:
    -------
    arr: NumPy array to be scaled to [new_min, new_max] range
    new_range: tuple of form (min, max) specifying range of
        transformed array

    Returns:
    -------
    NumPy array that has been scaled to be in
    [new_range[0], new_range[1]] range
    """
    # get array's current min and max
    mn = arr.min()
    mx = arr.max()

    # check if scaling needs to be done to be in new_range
    if mn < new_range[0] or mx > new_range[1]:
        # perform min-max scaling
        scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
    else:
        # return array if already in range
        scaled = arr

    return scaled

def _scale_array(arr, clip=True):
    """
    Trim NumPy array values to be in [0, 255] range with option of
    clipping or scaling.

    Parameters:
    -------
    arr: array to be trimmed to [0, 255] range
    clip: should array be scaled by np.clip? if False then input
        array will be min-max scaled to range
        [max([arr.min(), 0]), min([arr.max(), 255])]

    Returns:
    -------
    NumPy array that has been scaled to be in [0, 255] range
    """
    if clip:
        scaled = np.clip(arr, 0, 255)
    else:
        scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
        scaled = _min_max_scale(arr, new_range=scale_range)

    return scaled

@st.cache_data
def open_uploaded_image(file:UploadedFile) -> NDArray:
    image = Image.open(file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image 

@st.cache_data
def load_origin():
    target = cv2.imread("origin.png")
    return target

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg', "raw"])
processed = None
if uploaded_file is not None:
    image = open_uploaded_image(uploaded_file)
    if os.path.exists('origin.png'):
        st.markdown("Color map found!")
        source = load_origin()
        processed = color_transfer(source, image, depth_model=Midas())
        st.session_state.show_button = True 
    else:
        st.markdown("Color map not found, \
            make sure to press 'Save colormap' button on the first page")

        st.session_state.show_button = False
col1, col2 = st.columns([0.5, 0.5])
with col1:
    if st.button("Show results:", disabled=not st.session_state.show_button):
        st.image(processed, width=300)
with col2:
    if st.session_state.export_data is not None:
        st.download_button("Export color map:",
                       file_name="params.json", 
                       data=json.dumps(st.session_state.export_data),
                       disabled=not st.session_state.show_button)