
import os
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

# actual model
from keras.applications.resnet50 import ResNet50

from PIL import Image
import time


root = os.path.abspath(".")
print(root + "")

# pick some image here, I put a few nice ones in a seperate directory
imagepath = root + "/images/space_shuttle.jpg"


IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

model = ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
model.trainable = False


def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image


def preprocessing(data, target):

    preprocessed_data = prepare_image(data, target=target)

    return preprocessed_data


image = Image.open(imagepath)

model_inp = preprocessing(image, target=(IMAGE_WIDTH, IMAGE_HEIGHT))

predictions = model.predict(model_inp)
results = imagenet_utils.decode_predictions(predictions)
print(results)


# Saving to files if you may need it
cwd = os.path.abspath('.')
model_dir = cwd + "/data/"

model_json = model.to_json()
with open(model_dir + "resnet50_imagenet.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights(model_dir + "resnet50_imagenet_weights.h5")
