import streamlit as st
import cv2
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from streamlit.runtime.uploaded_file_manager import UploadedFile
from typing import Tuple

@st.cache_data
def open_uploaded_image(file:UploadedFile) -> Tuple[NDArray, Tuple]:
    image = Image.open(file)
    image = np.array(image)
    image_luma =cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32") 
    stats = image_stats(image_luma)
    l, _, a, _, b, _ = stats
    l = scale_stats(l)
    a = scale_stats(a)
    b = scale_stats(b)
    return image, (l, a, b)

@st.cache_data
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

html_insert = lambda x:f'<p style="text-align: center;">{x}</p>'

def scale_stats(value):
    return int((value/255) * 100)

def back_scale_stats(value):
    return (float(value)/100)*255

def form_image(image, luma_value, a_value,b_value):
    luma_value = back_scale_stats(luma_value)
    a_value = back_scale_stats(a_value)
    b_value = back_scale_stats(b_value)
    image =cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype("float32") 
    (l, a, b) = cv2.split(image)
    if not np.isclose(l.mean(), luma_value, rtol=0.1):
        l = l + (luma_value - l.mean())

        print(luma_value, l.mean())

    if not np.isclose(a.mean(), a_value, rtol=0.1):
        a = a + (a_value - a.mean())
    
    if not np.isclose(b.mean(), b_value, rtol=0.1):
        b = b+ (b_value - b.mean())

    new_image = cv2.merge([l, a, b])
    new_image = cv2.cvtColor(new_image.astype("uint8"), cv2.COLOR_LAB2BGR)
    return new_image
def create_sliders_callback(): 
    st.session_state['luma'] = st.session_state.init_l
    st.session_state['alpha_slider'] = st.session_state.init_a
    st.session_state['beta_slider'] = st.session_state.init_b

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg', "raw"])
if uploaded_file is not None:
    image, stats = open_uploaded_image(uploaded_file)
    init_l, init_b, init_a = stats
    if "init_l" not in st.session_state:
        st.session_state.init_l = init_l
        st.session_state.init_b = init_a
        st.session_state.init_a = init_b
    col01, col02 = st.columns( [0.5, 0.5])
    with col01:
        l_channel=st.slider("luminance", 
                            min_value=0, 
                            max_value=100, 
                            value=init_l, 
                            step=2,
                            help='L channel', 
                            key='luma')
        a_channel=st.slider("alpha", 
                            min_value=0, 
                            max_value=100, 
                            value=init_a, 
                            step=2,
                            help='A channel', 
                            key='alpha_slider')
        b_channel=st.slider("beta", 
                            min_value=0, 
                            max_value=100, 
                            value=init_b, 
                            step=2,
                            help='B channel', 
                            key="beta_slider")
    with col02:
        st.button('Reset colors', on_click=create_sliders_callback)
            
    processed_image = image.copy()
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown(html_insert("Before"),unsafe_allow_html=True)
        st.image(image,width=300)  

    with col2:
        st.markdown(html_insert("After"),unsafe_allow_html=True)
        processed_image = form_image(processed_image, l_channel, a_channel, b_channel)
        st.image(processed_image, width=300)
        if st.button("Save current color profile:"):
            cv2.imwrite("./origin.png", processed_image)
            st.write("Color profile saved to ./origin.png")

class ImageProcessor:
    def __init__(self, image):
        self.origin = image
        self.luma_origin =cv2.cvtColor(self.origin, cv2.COLOR_BGR2LAB).astype("float32") 
        self.origin_stats = image_stats(self.origin)
    def scale_image(self, x):
        pass
