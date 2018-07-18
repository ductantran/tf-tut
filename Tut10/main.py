import matplotlib.pyplot as plt
import PIL
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import knifey

from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Dropout
from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop


# Helper function for joining a directory and list of filenames
def path_join(dirname, filenames):
    return [os.path.join(dirname, filename) for filename in filenames]


# Helper function for plotting image
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i],
                      interpolation=interpolation)

            cls_true_name = class_names[cls_true[i]]

            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# Helper function for plotting confusion matrix
def print_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,
                          y_pred=cls_pred)

    print("Confusion matrix:")

    print(cm)

    for i, class_name in enumerate(class_names):
        print("({0}) {1}".format(i, class_name))


# Helper function for plotting example errors
def plot_example_errors(cls_pred):
    incorrect = (cls_pred != cls_test)

    image_paths = np.array(image_paths_test)[incorrect]

    images = load_images(image_paths=image_paths[0:9])

    cls_pred = cls_pred[incorrect]

    cls_true = cls_test[incorrect]

    plot_images(images=images,
                cls_true=cls_true[0:9],
                cls_pred=cls_pred[0:9])


def example_errors():
    generator_test.reset()

    y_pred = new_model.predict_generator(generator=generator_test, steps=steps_test)

    cls_pred = np.argmax(y_pred, axis=1)

    plot_example_errors(cls_pred)

    print_confusion_matrix(cls_pred)


# Helper function for loading images
def load_images(image_paths):
    images = [plt.imread(path) for path in image_paths]

    return np.asarray(images)


# Helper function for plotting training history
def plot_training_history(history):
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']

    val_acc = history.history['val_categorical_accuracy']
    val_loss = history.history['val_loss']

    plt.plot(acc, linestyle='-', color='b', label='Training Acc.')
    plt.plot(loss, 'o', color='b', label='Training Loss')

    plt.plot(val_acc, linestyle='--', color='r', label='Test Acc.')
    plt.plot(val_loss, 'o', color='r', label='Test Loss')

    plt.title('Training and Test Accuracy')
    plt.legend()

    plt.show()


# DATASET: KNIFE-SPOONY #
knifey.maybe_download_and_extract()
knifey.copy_files()

train_dir = knifey.train_dir
test_dir = knifey.test_dir

# PRE-TRAINED MODEL: VGG16 #
model = VGG16(include_top=True, weights='imagenet')

# INPUT PIPELINE #
input_shape = model.layers[0].output_shape[1:3]

# Augmented Dataset
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=180,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=[0.9, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

datagen_test = ImageDataGenerator(rescale=1./255)

batch_size = 20

if True:
    save_to_dir = None
else:
    save_to_dir = 'augmented_images/'

generator_train = datagen_train.flow_from_directory(directory=train_dir, target_size=input_shape, batch_size=batch_size, shuffle=True, save_to_dir=save_to_dir)
generator_test = datagen_test.flow_from_directory(directory=test_dir, target_size=input_shape, batch_size=batch_size, shuffle=False)

steps_test = generator_test.n / batch_size

image_paths_train = path_join(train_dir, generator_train.filenames)
image_paths_test = path_join(test_dir, generator_test.filenames)

cls_train = generator_train.classes
cls_test = generator_test.classes

class_names = list(generator_train.class_indices.keys())

num_classes = generator_train.num_classes

# Plot a few images to see if data is correct
images = load_images(image_paths_train[:9])
cls_true = cls_train[:9]
plot_images(images, cls_true)

# CLASS WEIGHTS #
class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(cls_train), y=cls_train)


# EXAMPLE PREDICTIONS #
def predict(image_path):
    img = PIL.Image.open(image_path)
    img_resized = img.resize(input_shape, PIL.Image.LANCZOS)

    plt.imshow(img_resized)
    plt.show()

    img_array = np.expand_dims(np.array(img_resized), axis=0)

    pred = model.predict(img_array)

    pred_decoded = decode_predictions(pred)[0]

    for code, name, score in pred_decoded:
        print('{0:>6.2%} : {1}'.format(score, name))


predict(image_paths_train[28])

# TRANSFER LEARNING #
model.summary()

transfer_layer = model.get_layer('block5_pool')

conv_model = Model(inputs=model.input, outputs=transfer_layer.output)

new_model = Sequential()
new_model.add(conv_model)
new_model.add(Flatten())
new_model.add(Dense(1024, activation='relu'))
new_model.add(Dropout(0.5))
new_model.add(Dense(num_classes, activation='softmax'))

optimizer = Adam(lr=1e-5)

loss = 'categorical_crossentropy'

metrics = ['categorical_accuracy']


def print_layer_trainable():
    for layer in conv_model.layers:
        print('{0}:\t{1}'.format(layer.trainable, layer.name))


print('TRANSFER LEARNING')
conv_model.trainable = False
for layer in conv_model.layers:
    layer.trainable = False
print_layer_trainable()

new_model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

epochs = 5
steps_per_epoch = 5

history = new_model.fit_generator(
    generator=generator_train,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weight,
    validation_data=generator_test,
    validation_steps=steps_test
)

plot_training_history(history)

result = new_model.evaluate_generator(generator=generator_test, steps=steps_test)
print('Test-set Classification Accuracy: {0:.2%}'.format(result[1]))

example_errors()

# FINE-TUNING #
print('Fine Tuning')
conv_model.trainable = True
for layer in conv_model.layers:
    layer.trainable = ('block5' in layer.name or 'block4' in layer.name)
print_layer_trainable()

optimizer_fine = Adam(lr=1e-7)

new_model.compile(optimizer=optimizer_fine, loss=loss, metrics=metrics)

history = new_model.fit_generator(
    generator=generator_train,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    class_weight=class_weight,
    validation_data=generator_test,
    validation_steps=steps_test
)

plot_training_history(history)

result = new_model.evaluate_generator(generator=generator_test, steps=steps_test)
print('Test-set Classification Accuracy: {0:.2%}'.format(result[1]))

example_errors()
