import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="About This App",
    page_icon="ℹ️", 
)

title_alignment="""
    <style>
    p, h3, h1, h5, h6 {
    text-align: center
    }
    div:has(> img){
        width: 100% !important  
    }
    img{
        align-self: center
    }
    </style>
    """
st.markdown(title_alignment, unsafe_allow_html=True)

st.write("# NASA Mars Colorized")

st.header("")

st.markdown(
    """
    This app is a simple web app that retrieve Mars rover photos from [NASA's API](https://api.nasa.gov/). Unfortunately most of the raw photos from NASA are grayscale (black and white). This app can turn grayscale photos from NASA's rover into beautifull colorized photos. This app is powered by Generative Adversarial Network (GAN) model called [Pix2pix](https://arxiv.org/abs/1611.07004) founded by Phillip Isola to magically turn black-and-white photos into wonderfull colorized photos. The model was trained using [this notebook](https://www.kaggle.com/code/varunnagpalspyz/pix2pix-is-all-you-need) by [VARUN NAGPAL SPYZ](https://www.kaggle.com/varunnagpalspyz). You can also download raw and processed images easily using this app. Explore and enjoy Mars photos with this app!🎉
"""
)

st.header("")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    col_s1.image(Image.open("cache/118330.jpg"), width=150)
with col_s2:
    col_s2.image(Image.open("cache/48451.jpg"), width=150)
with col_s3:
    col_s3.image(Image.open("cache/286352.jpg"), width=150)
col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    col_p1.image(Image.open("cache/processed/118330.jpg"), width=150)
with col_p2:
    col_p2.image(Image.open("cache/processed/48451.jpg"), width=150)
with col_p3:
    col_p3.image(Image.open("cache/processed/286352.jpg"), width=150)

st.header("")

st.image(Image.open("imgs/nasa.png"), width=150)

st.header("")
st.header("")

st.write("##### Creator")
st.image(Image.open("imgs/alif.png"), width=150)
st.write("##### Nur Alif Ilyasa")
st.write("Universitas Negeri Yogyakarta")

st.header("")
st.header("")

st.write("##### Credits")
st.write("Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2016). Image-to-Image Translation with Conditional Adversarial Networks (Version 3). arXiv. https://doi.org/10.48550/ARXIV.1611.07004")
