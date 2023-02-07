import numpy as np
import matplotlib.pyplot as plt
import cv2

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

st.title("Coin Count App")
st.write("Judge \500")

minDist = st.slider("minDist", min_value=0, max_value=100, step=10, value=20)
param1 = st.slider("param1", min_value=0, max_value=1000, step=10, value=100)
param2 = st.slider("param2", min_value=0, max_value=1000, step=10, value=70)
minRadius = st.slider("minRadius", min_value=0, max_value=1000, step=10, value=30)
maxRadius = st.slider("maxRadius", min_value=0, max_value=1000, step=10, value=200)

def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def callback(frame):
    img = frame.to_ndarray(format="bgr24")
    # カラー画像として読み込む(cv2.IMREAD_GRAYSCALE指定なし)
    im = cv2.imread(img)
    # そのままだと画像サイズが大きすぎるので、resizeメソッドにて縦横比変えず縮小
    im = cv2.resize(im, dsize=None, fx=0.5, fy=0.5)
    # ここでグレースケール化、これをHoughCirclesメソッドの入力として使用
    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,
        dp=1.0, minDist=minDist, param1=param1, param2=param2,
        minRadius=minRadius, maxRadius=maxRadius)
    num = 0

    if circles is not None and len(circles) > 0:
        circles = np.uint16(np.around(circles)) # uint16型に変更
        for i in circles[0,:]:
            cv2.circle(im,(i[0],i[1]),i[2],(0,255,0),2) # 円周を描画(緑)
            cv2.circle(im,(i[0],i[1]),2,(0,0,255),2) # 中心点を描画(赤)
            num += 1 # 検知した円の数

    print(f"検知した円の数: {str(num)}")
    st.write(f"検知した円の数: {str(num)}")
    out_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return av.VideoFrame.from_ndarray(out_img, format="bgr24")

# Colabで動かす場合ここを追記
webrtc_streamer(
    key="example",
    video_frame_callback=callback,
    rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
