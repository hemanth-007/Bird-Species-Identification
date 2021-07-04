import os
import tensorflow as tf

class_names = []
dir = "Data/test"
class_names = os.listdir(dir)


def get_classes():
    return class_names


def load_and_prep(image, shape=224, scale=False):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, size=([shape, shape]))
    if scale:
        image = image/255.
    return image
