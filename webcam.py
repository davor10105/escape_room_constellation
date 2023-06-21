from re import search
import cv2
import numpy as np
import streamlit as st
from streamlit_extras.let_it_rain import rain
import math
import time
import pickle


def contours_from_image(image):
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image, bottom_thresh, top_thresh)

    kernel = np.ones(smoothing_kernel, np.uint8)

    image = cv2.erode(np.uint8(image), kernel, iterations=1)
    image = cv2.dilate(np.uint8(image), kernel, iterations=1)

    image = np.uint8(image)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def threshold_image(image):
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image, bottom_thresh, top_thresh)

    kernel = np.ones(smoothing_kernel, np.uint8)

    image = cv2.erode(np.uint8(image), kernel, iterations=1)
    image = cv2.dilate(np.uint8(image), kernel, iterations=1)

    image = np.uint8(image)

    return image


def get_iter_image():
    iter_image = np.zeros(image_size)
    for _ in range(num_starting_iters):
        ret, frame = vid.read()
        original_image = cv2.resize(frame, image_size)
        #original_image = original_image[:256, 256:]
        #original_image = cv2.resize(original_image, image_size)

        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
        image = cv2.inRange(image, bottom_thresh, top_thresh)

        kernel = np.ones(smoothing_kernel, np.uint8)

        image = cv2.erode(np.uint8(image), kernel, iterations=1)
        image = cv2.dilate(np.uint8(image), kernel, iterations=1)

        image = np.float32(image)
        iter_image += image
    iter_image /= num_starting_iters
    iter_image = np.where(iter_image < 65, 0, 255)
    iter_image = np.uint8(iter_image)

    return iter_image, original_image

def centroids_from_contours(contours):
    centroids = []
    for contour in contours:
        M = cv2.moments(contour)
        cX = int(M["m10"] / (M["m00"] + 1e-9))
        cY = int(M["m01"] / (M["m00"] + 1e-9))

        centroids.append([cX, cY])
    if len(centroids) > 0:
        centroids = np.stack(centroids)
    else:
        centroids = []

    return centroids


def draw_centroids(iter_image):
    contours, hierarchy = cv2.findContours(iter_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    centroids = centroids_from_contours(contours)
    centroid_image = np.zeros(image_size)
    for centroid in centroids:
        centroid_image = cv2.circle(centroid_image, (centroid[0], centroid[1]), 20, (255,), -1)

    return centroid_image


def find_starting_image():
    global starting_iter_image, starting_centroid_image
    starting_iter_image, _ = get_iter_image()
    starting_centroid_image = draw_centroids(starting_iter_image)

def reset():
    global vid
    vid.release()


#search_color = np.array([66, 99, 226])  # this is BGR (stupid cv2)
#search_color = np.array([41, 61, 62])
num_starting_iters = 5
smoothing_kernel = (5, 5)
distance_threshold = 0.1

#starting_iter_image = None
#starting_centroid_image = None

scan_started = False
start_time = time.time()

#search_color = search_color * np.array([0.5, 2.55, 2.55])
#color_threshold = np.array([5, 50, 50])

with open('saved_data.pickle', 'rb') as f:
    search_color, (bh, bs, bv), (th, ts, tv) = pickle.load(f)

bot_color_thresh = np.array([bh, bs, bv])
top_color_thresh = np.array([th, ts, tv])
bottom_thresh = search_color - bot_color_thresh
top_thresh = search_color + top_color_thresh
bottom_thresh = np.clip(bottom_thresh, a_min=0, a_max=None)
top_thresh = np.clip(top_thresh, a_min=None, a_max=255)
top_thresh[0] = min(179, top_thresh[0])
starting_threshold = None

image_size = (512, 512)

st.title('Travel :blue[destination] scanner :rocket:')
st.markdown('''
    #### Move away from the :red[camera] upon completing the pattern.
''')
progress_text = 'Ending destination completion progress bar'
progress_bar = st.progress(0, text=progress_text)

# define a video capture object
camera_window = st.image([])

if 'vid' not in st.session_state:
    vid = cv2.VideoCapture(0)
    st.session_state['vid'] = vid
else:
    vid = st.session_state['vid']

time.sleep(2)


refresh_button = st.button('.', on_click=lambda: reset)

#starting_iter_image, _ = get_iter_image()
#starting_centroid_image = draw_centroids(starting_iter_image)

with st.spinner('Saving constellation configuration...'):
    find_starting_image()

with st.expander("Refresh AKSADKMKDSKMDAKMDSAMDSAKM"):
    find_starting_image()

while (True):
    current_iter_image, original_image = get_iter_image()
    current_centroid_image = draw_centroids(current_iter_image)

    start_color_image = np.ones((image_size[0], image_size[1], 3)) * np.array([255, 0, 0]) * starting_centroid_image[:,:,None] / 255
    start_color_image = np.uint8(start_color_image)
    current_color_image = np.ones((image_size[0], image_size[1], 3)) * np.array([0, 0, 255]) * current_centroid_image[:,:,None] / 255
    current_color_image = np.uint8(current_color_image)

    show_image = np.where(start_color_image == 0, original_image, start_color_image)
    show_image = np.where(current_color_image == 0, show_image, current_color_image)

    distance = 1 - 2 * (starting_centroid_image / 255 * current_centroid_image / 255).sum() / ((starting_centroid_image / 255).sum() + (current_centroid_image / 255).sum())
    
    if math.isnan(distance):
        distance = 1.
    print(distance, time.time() - start_time)

    current_progress = 1 - distance if distance > distance_threshold else 1
    progress_bar.progress(current_progress, text=progress_text)

    if current_progress == 1 and time.time() - start_time > 10:
        rain(
            emoji="🎈",
            font_size=54,
            falling_speed=3,
            animation_length=1,
        )
        break

    camera_window.image(cv2.cvtColor(show_image, cv2.COLOR_BGR2RGB))

camera_window.image([])
vid.release()

st.title('Congratulations, the secret launch code is :red[5555] :sunglasses:')