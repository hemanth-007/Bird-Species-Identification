import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
from utils import load_and_prep, get_classes


@st.cache(suppress_st_warning=True)
def predict_species(image, model):
    image = load_and_prep(image)
    image = tf.cast(tf.expand_dims(image, axis=0), tf.int16)
    preds = model.predict(image)
    pred_class = class_names[tf.argmax(preds[0])]
    pred_conf = tf.reduce_max(preds[0])
    top_5_i = sorted((preds.argsort())[0][-5:][::-1])
    values = preds[0][top_5_i] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5_i[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df


class_names = get_classes()

st.set_page_config(page_title="Bird Species Identification",
                   page_icon="🐦")

#### SideBar ####

st.sidebar.title("Bird Species Identification")
st.sidebar.write("""
Bird Species identification is an end-to-end **CNN Image Classification Model** which identifies the bird species in your image. 
It can identify over 275 different bird species
It is based upon pre-trained Image Classification Models that comes with Keras and then retrained on the **Bird Species Dataset**.
""")
st.sidebar.write("""
**Dataset :** [**Bird Species**](https://www.kaggle.com/gpiosenka/100-bird-species)
""")
st.sidebar.write("""
**Model :** **`EfficientNetB1`**\n
**Accuracy :** **`95.56%`**\n
**Model :** **`InceptionNetV3`**\n
**Accuracy :** **`95.20%`**\n
**Model :** **`ResNet50`**\n
**Accuracy :** **`96.58%`**\n
**Model :** **`MobileNetV2`**\n
**Accuracy :** **`95.13%`**\n
""")
st.sidebar.markdown(body="""
<th style="border:None"><a href="https://www.linkedin.com/in/hemanth-mummidi-bba503203" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="Hemanth" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://github.com/hemanth-007" target="blank"><img align="center" src="https://bit.ly/3c2onZS" alt="Hemanth" height="40" width="40" /></a></th>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
Created by **Hemanth**
""")


#### Main Body ####


st.title("Bird Species 🦜🔍")
st.header("Identify your bird species!")
st.write(
    "To know more about this app, visit [**GitHub**](https://github.com/hemanth-007/Bird-Species-Identification)")

model = tf.keras.models.load_model("EfficientNetModel.hdf5")

file = st.file_uploader(label="Upload an image of bird.",
                        type=["jpg", "jpeg", "png"])

if not file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predict_species(image, model)
    st.success(
        f"""Prediction : {pred_class} | Confidence : {pred_conf*100:.2f}%""")
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=800, height=600))
