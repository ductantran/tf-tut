from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import math
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, concatenate
from tensorflow.python.keras.layers import Reshape, MaxPooling2D
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import to_categorical

# Configuration of Neural Network
filter_size_1 = 5
num_filters_1 = 16
filter_size_2 = 5
num_filters_2 = 36
fc_size = 128

# Load Data
data = input_data.read_data_sets('data/MNIST-dep/', one_hot=True)
print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))
data.test.cls = np.argmax(data.test.labels, axis=1)

# Data Dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
img_shape_full = (img_size, img_size, 1)
num_channels = 1
num_classes = 10


def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
# plot_images(images=images, cls_true=cls_true)


# Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Functional Model
inputs = Input(shape=(img_size_flat,))

net = inputs
net = Reshape(img_shape_full)(net)
net = Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv_1')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv_2')(net)
net = MaxPooling2D(pool_size=2, strides=2)(net)
net = Flatten()(net)
net = Dense(128, activation='relu')(net)
net = Dense(num_classes, activation='softmax')(net)

outputs = net

# # Functional Model
# inputs = Input(shape=(img_size_flat,))
#
# inp = inputs
# inp_tensor = Reshape(img_shape_full)(inp)
# conv_1 = Conv2D(kernel_size=5, strides=1, filters=16, padding='same', activation='relu', name='layer_conv_1')(inp_tensor)
# conv_1 = MaxPooling2D(pool_size=2, strides=2)(conv_1)
# conv_2 = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv_2')(conv_1)
# conv_2 = MaxPooling2D(pool_size=2, strides=2)(conv_2)
# conv_3 = Conv2D(kernel_size=5, strides=1, filters=36, padding='same', activation='relu', name='layer_conv_3')(inp_tensor)
# conv_3 = MaxPooling2D(pool_size=2, strides=2)(conv_3)
# conv_3 = MaxPooling2D(pool_size=2, strides=2)(conv_3)
#
# merged = concatenate([conv_2, conv_3], axis=1)
# net = Flatten()(merged)
# net = Dense(128, activation='relu')(net)
# net = Dense(num_classes, activation='softmax')(net)
#
# outputs = net

# Model Compilation
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(x=data.train.images, y=data.train.labels, epochs=1, batch_size=128)

# Evaluation
result = model.evaluate(x=data.test.images, y=data.test.labels)

for name, value in zip(model.metrics_names, result):
    print(name, value)

y_pred = model.predict(x=data.test.images)
print(y_pred)
cls_pred = np.argmax(y_pred, axis=1)


def plot_example_errors(cls_pred):
    incorrect = (cls_pred != data.test.cls)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]

    plot_images(images=images[0:9],
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


plot_example_errors(cls_pred)

path_model = 'model.keras'

model.save(path_model)

del model

model_2 = load_model(path_model)

# Prediction
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
y_pred = model_2.predict(x=images)
cls_pred = np.argmax(y_pred, axis=1)
plot_images(images=images, cls_true=cls_true, cls_pred=cls_pred)

model_2.summary()

layer_input = model_2.layers[0]
layer_conv_1 = model_2.layers[2]
layer_conv_2 = model_2.layers[4]


def plot_conv_output(values):
    num_filters = values.shape[3]
    num_grids = math.ceil(math.sqrt(num_filters))
    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = values[0, :, :, i]

            ax.imshow(img, interpolation='nearest', cmap='binary')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


image_1 = data.test.images[45]

output_conv_1 = K.function(inputs=[layer_input.input], outputs=[layer_conv_1.output])
layer_output_1 = output_conv_1([[image_1]])[0]
plot_conv_output(values=layer_output_1)

output_conv_2 = Model(inputs=layer_input.input, outputs=layer_conv_2.output)
layer_output_2 = output_conv_2.predict(np.array([image_1]))
plot_conv_output(values=layer_output_2)
