# Bird-Species-Identification 🦜🔍

Bird Species Identification is an end-to-end CNN Image Classification Model which identifies the bird species in an image. It can identify over 275 different bird species.

It is based upon pre-trained Image Classification Models that comes with Keras and then retrained on the [**Bird Species Dataset**](https://www.kaggle.com/gpiosenka/100-bird-species).

>**Model :** **`EfficientNetB1`**
>**Accuracy :** **`95.56%`**

>**Model :** **`InceptionNetV3`**
>**Accuracy :** **`95.20%`**

>**Model :** **`ResNet50`**
>**Accuracy :** **`96.58%`**

>**Model :** **`MobileNetV2`**
>**Accuracy :** **`95.13%`**

### To view the Deployed app, [**Click here**](https://birds-species-identification.herokuapp.com/)

> The app may take a couple of seconds to load for the first time, but it works perfectly fine.

![Screenshot](https://github.com/hemanth-007/Bird-Species-Identification/blob/main/Extras/s2.png)
![Screenshot](https://github.com/hemanth-007/Bird-Species-Identification/blob/main/Extras/s1.png)


## Breaking down the repo

* `.gitignore` : Tells which files/folders to ignore while tracking
* `app.py`  : Contains web app code built using [**streamlit**](https://streamlit.io/) api
* `utils.py`  : Some of used fuctions in  `app.py`
* `transfer_learning_model_training.ipynb`  : Jupyter Notebook used to train and evaluate Models
* `Models`  : Contains all the models in *.hfd5* format
* `requirements.txt`  : List of required dependencies
