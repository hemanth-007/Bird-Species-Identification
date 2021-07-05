# Bird-Species-Identification 🦜🔍

Bird Species Identification is an end-to-end CNN Image Classification Model which identifies the bird species in an image. It can identify over 275 different bird species.

It is based upon pre-trained Image Classification Models that comes with Keras and then retrained on the [**Bird Species Dataset**](https://www.kaggle.com/gpiosenka/100-bird-species).

I've trained the model on 4 different CNN Architectures to get an overview of how each network performs. 

>**Model :** **`EfficientNetB1`**
>**Accuracy :** **`95.56%`**

>**Model :** **`InceptionNetV3`**
>**Accuracy :** **`95.20%`**

>**Model :** **`ResNet50`**
>**Accuracy :** **`96.58%`**

>**Model :** **`MobileNetV2`**
>**Accuracy :** **`95.13%`**

After training the model, I've exported it in *.hfd5* format and then integrated it with the **streamlit Web Framework**

[**Streamlit**](https://streamlit.io/) is an open-source app framework that turns data scripts into shareable web apps in minutes.

Once I got the App running on my local environment, I then deployed the App
on the [**Heroku**](https://www.heroku.com) platform.

### To view the Deployed app [**Click here**](https://birds-species-identification.herokuapp.com/)

> The app may take a couple of seconds to load for the first time, but it works perfectly fine.

![Screenshot](https://github.com/hemanth-007/Bird-Species-Identification/blob/main/Extras/s2.png)
![Screenshot](https://github.com/hemanth-007/Bird-Species-Identification/blob/main/Extras/s1.png)

> If you want to dive deeper on how the model was trained check out **[`transfer-learning-model-training.ipynb`](https://github.com/hemanth-007/Bird-Species-Identification/blob/main/Transfer_learning_model_training.ipynb) Notebook**

## Breaking down the repo

* `.gitignore` : Tells which files/folders to ignore while tracking
* `.slugignore` : Contains which files to be removed after you push code to Heroku and before the buildpack runs.
* `app.py`  : Contains web app code built using [**streamlit**](https://streamlit.io/) api
* `utils.py`  : Some of used fuctions in  `app.py`
* `transfer_learning_model_training.ipynb`  : Jupyter Notebook used to train and evaluate Models
* `Models`  : Contains all the models in *.hfd5* format
* `requirements.txt`  : List of required dependencies
