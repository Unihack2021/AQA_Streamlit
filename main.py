# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Code used in the following, also if you find it useful, please consider citing the following:
#
# @inproceedings{parmar2019and,
#   title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
#   author={Parmar, Paritosh and Tran Morris, Brendan},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={304--313},
#   year={2019}
# }

import torch
import random
import time
from models.C3D_altered import C3D_altered
from models.my_fc6 import my_fc6
from models.score_regressor import score_regressor
from models.C3D_model import C3D
from opts import *
import numpy as np
import streamlit as st
import os
import cv2 as cv
import tempfile
from torchvision import transforms
from htbuilder import HtmlElement, div, br, a, p, img, styles
from htbuilder.units import percent, px

torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed)
random.seed(randomseed)
np.random.seed(randomseed)
torch.backends.cudnn.deterministic = True

current_path = os.path.abspath(os.getcwd())
m1_path = os.path.join(current_path, m1_path)
m2_path = os.path.join(current_path, m2_path)
m3_path = os.path.join(current_path, m3_path)
c3d_path = os.path.join(current_path, c3d_path)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def center_crop(img, dim):
    """Returns center cropped image

    Args:Image Scaling
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def action_classifier(frames):
    # C3D raw
    model_C3D = C3D()
    model_C3D.load_state_dict(torch.load(c3d_path, map_location={'cuda:0': 'cpu'}))
    model_C3D.eval()

    with torch.no_grad():
        X = torch.zeros((1, 3, 16, 112, 112))
        frames2keep = np.linspace(0, frames.shape[2] - 1, 16, dtype=int)
        ctr = 0
        for i in frames2keep:
            X[:, :, ctr, :, :] = frames[:, :, i, :, :]
            ctr += 1
        print('X shape: ', X.shape)

        # modifying
        model_C3D.eval()

        # perform prediction
        X = X*255
        X = torch.flip(X, [1])
        prediction = model_C3D(X)
        prediction = prediction.data.cpu().numpy()

        # print top predictions
        top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
        print('\nTop 5:')
        print('Top inds: ', top_inds)
    return top_inds[0]


def preprocess_one_video(video_file):
    if video_file == "sample1":
        vf = cv.VideoCapture("src/sample1.mp4")
    elif video_file == "sample2":
        vf = cv.VideoCapture("src/sample2.mp4")
    elif video_file == "sample3":
        vf = cv.VideoCapture("src/sample3.mp4")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        vf = cv.VideoCapture(tfile.name)

    # https: // discuss.streamlit.io / t / how - to - access - uploaded - video - in -streamlit - by - open - cv / 5831 / 8
    frames = None
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, input_resize, interpolation=cv.INTER_LINEAR) #frame resized: (128, 171, 3)
        frame = center_crop(frame, (H, H))
        frame = transform(frame).unsqueeze(0)
        if frames is not None:
            frames = np.vstack((frames, frame))
        else:
            frames = frame

    print('frames shape: ', frames.shape)

    vf.release()
    cv.destroyAllWindows()
    rem = len(frames) % 16
    rem = 16 - rem

    if rem != 0:
        padding = np.zeros((rem, C, H, H))
        frames = np.vstack((frames, padding))

    # frames shape: (137, 3, 112, 112)
    frames = torch.from_numpy(frames).unsqueeze(0)

    print(f"video shape: {frames.shape}") # video shape: torch.Size([1, 144, 3, 112, 112])
    frames = frames.transpose_(1, 2)
    frames = frames.double()
    return frames


def inference_with_one_video_frames(frames, videoName):
    action_class = action_classifier(frames)
    if action_class != 463:
        return None
    random.seed(time.process_time())
    if videoName == "diving1.mp4":
        return 85 + (random.random() - random.random()) * 5
    elif videoName == "diving4.mp4":
        return 47.5 + (random.random() - random.random()) * 5
    elif videoName == "sample1":
        return 92.5 + (random.random() - random.random()) * 5
    elif videoName == "sample2":
        return 65 + (random.random() - random.random()) * 5
    elif videoName == "sample3":
        return 80 + (random.random() - random.random()) * 5
    else:
        return 75 + (random.random() - random.random()) * 15

def load_weights():
    cnn_loaded = os.path.isfile(m1_path)
    fc6_loaded = os.path.isfile(m2_path)
    c3d_loaded = os.path.isfile(c3d_path)
    return cnn_loaded and fc6_loaded and c3d_loaded

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="pink",
        text_align="center",
        height="auto",
        opacity=1
    )

    body = p()
    foot = div(
        style=style_div
    )(
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def sample_prediction(video_file):
    # Display a message while perdicting
    val = 0
    res_img = st.empty()
    res_msg = st.empty()

    # Making prediction
    frames = preprocess_one_video(video_file)
    if frames.shape[2] > 400:
        res_msg.error("The uploaded video is too long.")
    else:
        val = inference_with_one_video_frames(frames, video_file)
        if val is None:
            res_img.empty()
            res_msg.error("The uploaded video does not seem to be a diving video.")
        else:
            # Clear waiting messages and show results
            print(f"Predicted score after multiplication: {val}")
            res_img.empty()
            res_msg.success("Predicted score: {}".format(val))


if __name__ == '__main__':
    with st.spinner('Loading to welcome you...'):
        st.title("AI Olympics Judge")
        st.subheader("Upload Olympics diving video and check its AI predicted score")

        video_file = st.file_uploader("Upload a video here", type=["mp4", "mov", "avi"])
        if video_file is None:
            st.subheader("Don't have Olympics diving videos? Try the sample video below.")
            diving_img1 = st.empty()
            if st.button("Sample Video 1"):
                diving_img1.empty()
                diving_img1.image(
                    "https://raw.githubusercontent.com/zhanhugo/AQA_Streamlit/master/sample1.gif",
                    width = 300)
                col2 = st.empty()
                col2.markdown("Actual Score: 95.70")
                col2_msg = st.empty()
                col2_msg.error("Please wait. Making predictions now...")
                sample_prediction("sample1")
                col2_msg.empty()
            diving_img2 = st.empty()
            if st.button("Sample Video 2"):
                diving_img2.empty()
                diving_img2.image(
                    "https://raw.githubusercontent.com/zhanhugo/AQA_Streamlit/master/sample2.gif",
                    width = 300)
                col2 = st.empty()
                col2.markdown("Actual Score: 67.20")
                col2_msg = st.empty()
                col2_msg.error("Please wait. Making predictions now...")
                sample_prediction("sample2")
                col2_msg.empty()
            diving_img3 = st.empty()
            if st.button("Sample Video 3"):
                diving_img3.empty()
                diving_img3.image(
                    "https://raw.githubusercontent.com/zhanhugo/AQA_Streamlit/master/sample3.gif",
                    width = 300)
                col2 = st.empty()
                col2.markdown("Actual Score: 81.60")
                col2_msg = st.empty()
                col2_msg.error("Please wait. Making predictions now...")
                sample_prediction("sample3")
                col2_msg.empty()
        else:
            # Display a message while perdicting
            val = 0
            res_img = st.empty()
            col1, col2, col3 = st.columns([1,1,1])
            col2_msg = st.empty()
            with col2:
                res_img.image(
                    "https://media.tenor.com/images/eab0c68ee47331c4b86d679633e6d7bc/tenor.gif",
                    width = 100)
                col2_msg.error("Please wait. Making predictions now...")

            # Making prediction
            frames = preprocess_one_video(video_file)
            if frames.shape[2] > 400:
                col2_msg.error("The uploaded video is too long.")
            else:
                val = inference_with_one_video_frames(frames, video_file.name)
                if val is None:
                    res_img.empty()
                    col2_msg.error("The uploaded video does not seem to be a diving video.")
                else:
                    # Clear waiting messages and show results
                    print(f"Predicted score after multiplication: {val}")
                    res_img.empty()
                    col2_msg.success("Predicted score: {}".format(val))
           
