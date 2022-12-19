from PIL import Image
import streamlit as st
import requests
import json
import urllib
import urllib.request
from pathlib import Path
from os import path

import torch
from torchvision import transforms

from utils import build_backbone_unet, MainModel, device, Config, lab_to_rgb, save_img_from_np, get_downloadable_img

st.set_page_config(
    page_title="NASA Mars Rover's Colorized | Nur Alif Ilyasa",
    page_icon="🌍", 
)

@st.cache
def get_model():
    gen = build_backbone_unet(input_channels=1, output_channels=2, size=Config.image_size_2)

    gen.load_state_dict(torch.load("models/res18-unet.pt", map_location=device))
    return MainModel(generator=gen)

model = get_model()

title_alignment="""
    <style>
    p, h1, h3, h6, h4, .stDownloadButton {
    text-align: center
    }
    div:has(> img){
        width: 100% !important  
    }
    img{
        align-self: center
    }
    .stSpinner {

    }
    </style>
    """
st.markdown(title_alignment, unsafe_allow_html=True)

###

apikey = "uaTeTIp3KK6yKt94DbjwA4UbwN554oydFvJit2el"
# base_url = "https://api.nasa.gov/planetary/earth/assets?lon=-95.33&lat=29.78&date=2019-01-01&&dim=0.10&api_key="

def download_image(url, full_path):
    urllib.request.urlretrieve(url, full_path)
    # print("FILE DOWNLOADED")

def url_retrieve1(url, outfile):
    p = Path(outfile)
    if not p.is_file():
        R = requests.get(url, allow_redirects=True)
        if R.status_code != 200:
            raise ConnectionError('could not download {}\nerror code: {}'.format(url, R.status_code))
        p.write_bytes(R.content)

#

st.title("NASA Mars Rover's Photo Colorization")

rover = st.selectbox(
    'Wich Rover?',
    ('Spirit', 'Opportunity', 'Curiosity'))

cams = ['FHAZ', 'RHAZ', 'NAVCAM']
if rover == 'Curiosity':
    cams.append('MAST')
    cams.append('CHEMCAM')
    cams.append('MAHLI')
    cams.append('MARDI')
else:
    cams.append('PANCAM')
    cams.append('MINITES')

camera = st.selectbox(
    'Wich Camera?',
    cams)
sol = st.number_input('Sol (Days in Mars):', value=60)

#

base_url = "https://api.nasa.gov/mars-photos/api/v1/rovers/"+str(rover).lower()+"/photos?sol="+str(sol)+"&camera="+str(camera).lower()+"&api_key="
# print(base_url)
cachefolder = "cache/"

@st.cache
def fetch_data(url):
    x = requests.get(url)
    return json.loads(x.text)

obj = None
with st.spinner('Fetching data from NASA...'):
    obj = fetch_data(base_url+apikey)
    # print(obj)

photos = obj["photos"]

if len(photos) <= 0: 
    st.markdown("<h4>No photo available</h4><p>Try other options!</p>", unsafe_allow_html=True)
else : 
    with st.spinner('Photos available. Downloading photos...'):        
        for p in photos:
            url_retrieve1(p["img_src"], cachefolder + str(p["id"]) + ".jpg")


col1, col2 = st.columns(2)
with col1:
    st.markdown("<h4>Original Images</h4>", unsafe_allow_html=True)
with col2:
    st.markdown("<h4>Colorized Images</h4>", unsafe_allow_html=True)
    
for p in photos:
    with Image.open(cachefolder + str(p["id"]) + ".jpg") as im:
        col1, col2 = st.columns(2)
        
        im = im.convert('RGB')
        img_rszd = im
        img = transforms.ToTensor()(img_rszd)[:1] * 2. - 1.
        model.eval()
        with col1:
            st.image(img_rszd, width=300)
            im_todownload = get_downloadable_img(im, mul255=False)
            btn = st.download_button(
                label="Download Original Photo",
                data=im_todownload,
                file_name=str(p["id"])+"_original.png",
                mime="image/jpeg")
        with col2:
            with st.spinner('Processing Photo...'):
                s_path_processed = cachefolder+"processed/"+str(p["id"])+".jpg"
                p_path_processed = Path(s_path_processed)
                gen_output = None
                if not p_path_processed.is_file():
                    with torch.no_grad():
                        preds = model.generator(img.unsqueeze(0).to(device))
                        gen_output = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
                        save_img_from_np(gen_output, s_path_processed)

                        st.image(gen_output, width=300)
                        im_todownload = get_downloadable_img(gen_output)
                        btn = st.download_button(
                            label="Download Colorized Photo",
                            data=im_todownload,
                            file_name=str(p["id"])+"_colorized.png",
                            mime="image/jpeg")
                else:
                    # image already processed? skip processing
                    with Image.open(s_path_processed) as im:
                        gen_output = im

                        st.image(gen_output, width=300)
                        im_todownload = get_downloadable_img(gen_output, mul255=False)
                        btn = st.download_button(
                            label="Download Colorized Photo",
                            data=im_todownload,
                            file_name=str(p["id"])+"_colorized.png",
                            mime="image/jpeg")

