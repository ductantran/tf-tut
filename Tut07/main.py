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


# PANDA #
image_path = os.path.join(inception.data_dir, 'cropped_panda.jpg')
classify(image_path)

# INTERPRETATION OF CLASSIFICATION SCORES #
# Parrot (Original image)
classify('images/parrot.jpg')


# Helper function for plotting resized images
def plot_resized_image(image_path):
    resized_image = model.get_resized_image(image_path=image_path)
    plt.imshow(resized_image, interpolation='nearest')
    plt.show()


plot_resized_image(image_path='images/parrot.jpg')

# Parrot (Cropped Image, Top)
classify('images/parrot_cropped1.jpg')

# Parrot (Cropped Image, Middle)
classify('images/parrot_cropped2.jpg')

# Parrot (Cropped Image, Bottom)
classify('images/parrot_cropped3.jpg')

# Parrot (Padded)
classify('images/parrot_padded.jpg')

# Elon Musk (299x299)
classify('images/elon_musk.jpg')

# Elon Musk (100x100)
classify('images/elon_musk_100x100.jpg')
plot_resized_image(image_path='images/elon_musk_100x100.jpg')

# Willy Wonka (Gene Wilder)
classify('images/willy_wonka_old.jpg')

# Willy Wonka (Johnny Depp)
classify('images/willy_wonka_new.jpg')

model.close()
