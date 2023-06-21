import cv2
import numpy as np
import streamlit as st
from streamlit_extras.image_coordinates import streamlit_image_coordinates
import math
import time
from PIL import Image
import pickle


def save_to_file():
    with open('saved_data.pickle', 'wb') as f:
        pickle.dump((chosen_color, (bot_h, bot_s, bot_v), (top_h, top_s, top_v)), f)


# define a video capture object
if 'chosen_color' not in st.session_state:
    chosen_color = [0, 0, 0]
else:
    chosen_color = st.session_state['chosen_color']
print(chosen_color)

bot_h = st.slider('Bot Hue', 0, 179, 10, 5)
bot_s = st.slider('Bot Saturation', 0, 255, 35, 5)
bot_v = st.slider('Bot Value', 0, 255, 35, 5)

top_h = st.slider('Top Hue', 0, 179, 10, 5)
top_s = st.slider('Top Saturation', 0, 255, 35, 5)
top_v = st.slider('Top Value', 0, 255, 35, 5)

print(chosen_color)

if 'vid' not in st.session_state:
    vid = cv2.VideoCapture(0)
    st.session_state['vid'] = vid
else:
    vid = st.session_state['vid']

ret = False
while not ret:
    ret, frame = vid.read()

show_image = cv2.resize(frame, (512, 512))

show_image = cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB)
show_image = Image.fromarray(show_image)

coords = streamlit_image_coordinates(show_image, key='pil')
thresh_image = cv2.cvtColor(np.uint8(show_image), cv2.COLOR_RGB2HSV)
thresh_image = cv2.inRange(thresh_image, np.array(chosen_color) - np.array([bot_h, bot_s, bot_v]), np.array(chosen_color) + np.array([top_h, top_s, top_v]))
st.image(thresh_image)
st.button('Save', on_click=save_to_file)
if coords is not None:
    chosen_color = np.uint8(show_image)[coords['y'], coords['x']]
    chosen_color = cv2.cvtColor(chosen_color[None, None, :], cv2.COLOR_RGB2HSV)[0, 0]
    chosen_color = chosen_color.astype(int)
    st.session_state['chosen_color'] = chosen_color
    st.write(chosen_color)
    print(chosen_color)
    #vid.release()