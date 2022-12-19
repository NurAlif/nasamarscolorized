import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import requests
import json
import urllib
import urllib.request
from pathlib import Path
from os import path

title_alignment="""
    <style>
    p, h1, h3, h6, h4 {
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

###

loaded_model = tf.keras.models.load_model('my_model2.h5')

apikey = "uaTeTIp3KK6yKt94DbjwA4UbwN554oydFvJit2el"
# base_url = "https://api.nasa.gov/planetary/earth/assets?lon=-95.33&lat=29.78&date=2019-01-01&&dim=0.10&api_key="

def download_image(url, full_path):
    urllib.request.urlretrieve(url, full_path)
    print("FILE DOWNLOADED")

def url_retrieve1(url, outfile):
    p = Path(outfile)
    if not p.is_file():
        R = requests.get(url, allow_redirects=True)
        if R.status_code != 200:
            raise ConnectionError('could not download {}\nerror code: {}'.format(url, R.status_code))
        p.write_bytes(R.content)

#

st.title("NASA, Image Enhanchement")

rover = st.selectbox(
    'Wich Rover?',
    ('Curiosity', 'Opportunity', 'Spirit'))

cams = ['FHAZ', 'RHAZ', 'NAVCAM']
if rover[0] == 'Curiosity':
    cams.append('MAST')
    cams.append('CHEMCAM')
    cams.append('MAHLI')
    cams.append('MARDI')
else:
    cams.append('PANCAM')
    cams.append('MINITES')

camera = st.selectbox(
    'Wich Rover?',
    cams)
sol = st.number_input('Sol (Days in Mars):', value=1000)

#

base_url = "https://api.nasa.gov/mars-photos/api/v1/rovers/"+str(rover).lower()+"/photos?sol="+str(sol)+"&camera="+str(camera).lower()+"&api_key="
print(base_url)
cachefolder = "cache/"

obj = None
with st.spinner('Fetching data from NASA...'):
    x = requests.get(base_url+apikey)
    obj = json.loads(x.text)
    print(obj)

photos = obj["photos"]

if len(photos) <= 0: 
    st.markdown("<h4>No photo available</h4><p>Try other options!</p>", unsafe_allow_html=True)
else : 
    with st.spinner('Photo available. Wait for it...'):        
        for p in photos:
            url_retrieve1(p["img_src"], cachefolder + str(p["id"]) + ".jpg")


col1, col2 = st.columns(2)
with col1:
    st.write("Original Images")
with col2:
    st.write("Colorized Images")
    
for p in photos:
    with Image.open(cachefolder + str(p["id"]) + ".jpg") as im:
        col1, col2 = st.columns(2)
        
        im = im.convert('RGB')
        img_rszd = tf.image.resize(
            im,
            [224,224]
        )
        img_rszd.numpy()

        with col1:
            res = Image.fromarray(np.uint8(img_rszd)).convert('RGB')
            st.image(res, width=300)
        with col2:
            res = loaded_model.predict(np.expand_dims(img_rszd, axis=0))
            res = Image.fromarray(np.uint8(res[0])).convert('RGB')
            st.image(res, width=300)