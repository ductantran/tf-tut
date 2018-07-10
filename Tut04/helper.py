import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import confusion_matrix


# Plotting images
def plot_images(images, grid, img_shape, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == grid * grid

    fig, axes = plt.subplots(grid, grid)
    fig.subplots_adjust(hspace=grid/10, wspace=grid/10)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        if cls_pred is None:
            x_label = "True: {0}".format(cls_true[i])
        else:
            x_label = "True: {0}, Pred.: {1}".format(cls_true[i], cls_pred[i])

        ax.set_xlabel(x_label)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# Plotting example errors
def plot_example_errors(data, cls_pred, correct):
    incorrect = (correct == False)
    images = data.test.images[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]

    plot_images(images=images[:16], grid=4, img_shape=(28, 28), cls_true=cls_true[:16], cls_pred=cls_pred[:16])


# Plotting confusion matrix
def plot_confusion_matrix(cls_pred, cls_true, num_classes):
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)

    plt.matshow(cm)

    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.show()


# Plotting convolutional weights
def plot_conv_weights(session, weights, input_channel=0):
    w = session.run(weights)

    print("Mean: {0:.5f}, Stdev: {1:.5f}".format(w.mean(), w.std()))

    w_min = np.min(w)
    w_max = np.max(w)

    num_filters = w.shape[3]

    num_grids = math.ceil(math.sqrt(num_filters))

    fig, axes = plt.subplots(num_grids, num_grids)

    for i, ax in enumerate(axes.flat):
        if i < num_filters:
            img = w[:, :, input_channel, i]

            ax.imshow(img, vmin=w_min, vmax=w_max,
                      interpolation='nearest', cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
