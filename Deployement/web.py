# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 10:26:42 2023

@author: harin
"""

import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# Set up paths and load the model
model_path = "C:/Users/harin/Downloads/model (1).h5"
model = load_model(model_path)

# Define class names
class_names = ['Apple Braeburn', 'Apple Granny Smith', 'Apricot', 'Avocado', 'Banana',
               'Blueberry', 'Cactus fruit', 'Cantaloupe', 'Cherry', 'Clementine',
               'Corn', 'Cucumber Ripe', 'Grape Blue', 'Kiwi', 'Lemon', 'Limes',
               'Mango', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit',
               'Peach', 'Pear', 'Pepper Green', 'Pepper Red', 'Pineapple',
               'Plum', 'Pomegranate', 'Potato Red', 'Raspberry', 'Strawberry',
               'Tomato', 'Watermelon']

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.shutterstock.com/image-photo/blurred-mountain-background-260nw-269981756.jpg");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
add_bg_from_url()

# Define prediction function
def predict(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (100, 100))
    img_array = np.array(img_resized)
    img_expanded = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_expanded / 255.0
    prediction = model.predict(img_preprocessed)
    return prediction

# Define Streamlit app
def app():
    st.title("𝐎𝐁𝐉𝐄𝐂𝐓 𝐃𝐄𝐓𝐄𝐂𝐓𝐈𝐎𝐍")
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    # Upload image
    uploaded_file = st.file_uploader("𝐂𝐇𝐎𝐎𝐒𝐄 𝐀𝐍 𝐈𝐌𝐀𝐆𝐄", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("𝐂𝐋𝐀𝐒𝐒𝐈𝐅𝐈𝐄𝐃")
        prediction = predict(image)
        predicted_class = class_names[np.argmax(prediction)]
        st.write("Predicted Class:", predicted_class)

if __name__ == "__main__":
    app()
