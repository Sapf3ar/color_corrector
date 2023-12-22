import streamlit as st
import cv2
from PIL import Image
import numpy as np
from numpy.typing import NDArray
from streamlit.runtime.uploaded_file_manager import UploadedFile

@st.cache_data
def open_uploaded_image(file:UploadedFile) -> NDArray:
    image = Image.open(file)
    image = np.array(image)
    return image

html_insert = lambda x:f'<p style="text-align: center;">{x}</p>'

uploaded_file = st.file_uploader("", type=['jpg','png','jpeg', "raw"])
if uploaded_file is not None:
    image = open_uploaded_image(uploaded_file)
 
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown(html_insert("Before"),unsafe_allow_html=True)
        st.image(image,width=300)  

    with col2:
        st.markdown(html_insert("After"),unsafe_allow_html=True)
