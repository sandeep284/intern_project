import streamlit as st
import pickle
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Intern_project",layout="wide")
st.header("Potato Disease Prediction")
device=pickle.load(open("using.pkl",'rb'))
file = st.file_uploader("Please upload the leaf image of the potato", type=["jpg", "png"])

classes=['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict(model,img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = classes[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


if file is None:
    st.text("please upload the image...")
else:
    image = Image.open(file)
    st.image(image,use_column_width=True)
    x,y=predict(device,image)
    st.text(f"It is affected by {x}")
    st.text(f"The prediction is {y} sure")

