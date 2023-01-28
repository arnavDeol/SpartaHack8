import tensorflow as tf
model = tf.keras.models.load_model('')
import streamlit as st
import pandas as pd
import numpy as np
import cv2

#getting the image file from the user
st.write("""
         # Pothole(s) in Road Prediction
         """
         )
st.write("This is a simple image classification web app to predict the location of potholes in an image of a road")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


#processing the image submitted by the user
from PIL import Image, ImageOps
def import_and_predict(image_data, model):
    
        size = (150,150)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    #Change for potholes
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("There are _ potholes")
    elif np.argmax(prediction) == 1:
        st.write("There are no potholes")

    
    st.text("Probability (0: Paper, 1: Rock, 2: Scissor")
    st.write(prediction)


#streamlit run potholes.py