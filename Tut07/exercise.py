import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import inception
from PIL import Image

# DOWNLOAD THE INCEPTION MODEL #
inception.maybe_download()

# LOAD THE INCEPTION MODEL #
model = inception.Inception()


# Helper function for classifying and plotting images
def classify(image_path):
    pred = model.classify(image_path=image_path)

    model.print_scores(pred=pred, k=10, only_first_name=True)

    plt.imshow(Image.open(image_path), interpolation='nearest')
    plt.show()


# INTERPRETATION OF CLASSIFICATION SCORES #
# Tennis Ball (Original image)
classify('images/tennis_ball.jpg')

model.close()
