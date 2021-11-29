import streamlit as st
from PIL import Image
import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">"Image color clustering.."</h2>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.text("by hrishikesh jadhav")
st.image('logo.jpg', caption='INPUT',width=400)
st.image('logo_processed.jpeg', caption='OUTPUT',width=400)
st.header("we will use kmeans clustering algorithm")
st.write("First choose number of clusters you want from sidebar.")
st.text("Then upload a image for image color clustering..")
image_file = st.file_uploader("Upload An Image here",type=['png','jpeg','jpg'])

params = dict()
K = st.sidebar.slider('How  many clusters you want..?', 1, 10)
params['K'] = K


def image_segmentation(image):
    image_as_array = mpimg.imread(image)
    (h, w, c) = image_as_array.shape
    image_as_array2d = image_as_array.reshape(h*w,c)
    model = KMeans(n_clusters=params['K'])
    labels = model.fit_predict(image_as_array2d)
    rgb_codes = model.cluster_centers_.round(0).astype(int)
    quantized_image = np.reshape(rgb_codes[labels], (h, w, c))
    return quantized_image


def main():
    if image_file is not None:
        img = Image.open(image_file)
        st.image(img)
        with open(os.path.join("static", image_file.name),"wb") as f:
            f.write(image_file.getbuffer())
        data = image_segmentation("static/"+image_file.name)
        st.write("Clustering.............")
        blue, green, red = data.T
        data = np.array([red, green, blue])
        data = data.transpose()
        cv2.imwrite("static/data.jpg",data)
        img = Image.open("static/data.jpg")
        st.image(img)
        st.write("This is your clustered image....")

if __name__=='__main__':
    main()
