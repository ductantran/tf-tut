import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.metrics import confusion_matrix


# Plotting images
def plot_images(images,
                cls_true,
                ensemble_cls_pred=None,
                best_cls_pred=None):

    assert len(images) == len(cls_true)

    fig, axes = plt.subplots(3, 3)

    if ensemble_cls_pred is None:
        hspace = 0.3
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i].reshape((28, 28)), cmap='binary')

            if ensemble_cls_pred is None:
                xlabel = "True: {0}".format(cls_true[i])
            else:
                msg = "True: {0}\nEnsemble: {1}\nBest Net: {2}"
                xlabel = msg.format(cls_true[i],
                                    ensemble_cls_pred[i],
                                    best_cls_pred[i])

            ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
