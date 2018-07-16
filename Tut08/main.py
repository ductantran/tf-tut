import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import inception
from inception import transfer_values_cache
import prettytensor as pt
import cifar10
from cifar10 import num_classes
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# LOAD DATA FOR CIFAR-10 #
cifar10.maybe_download_and_extract()
class_names = cifar10.load_class_names()

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))


# Helper function for plotting images
def plot_images(images, cls_true, cls_pred=None, smooth=True):
    assert len(images) == len(cls_true) == 16

    fig, axes = plt.subplots(4, 4)

    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if smooth:
            interpolation = 'spline16'
        else:
            interpolation = 'nearest'

        ax.imshow(images[i, :, :, :], interpolation=interpolation)

        cls_true_name = class_names[cls_true[i]]

        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true_name)
        else:
            cls_pred_name = class_names[cls_pred[i]]
            xlabel = 'True: {0}\nPred: {1}'.format(cls_true_name, cls_pred_name)

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# PLOT A FEW IMAGES #
images = images_test[:16]
cls_true = cls_test[:16]
plot_images(images, cls_true, smooth=True)

# DOWNLOAD THE INCEPTION MODEL #
inception.maybe_download()

# LOAD THE INCEPTION MODEL #
model = inception.Inception()

# CALCULATE TRANSFER-VALUES #
file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')

print('Processing Inception transfer-values for training images...')
images_scaled = images_train * 255.0
transfer_values_train = transfer_values_cache(
    cache_path=file_path_cache_train,
    images=images_scaled,
    model=model)

print('Processing Inception transfer-values for test images...')
images_scaled = images_test * 255.0
transfer_values_test = transfer_values_cache(
    cache_path=file_path_cache_test,
    images=images_scaled,
    model=model)

print(transfer_values_train.shape, '|', transfer_values_test.shape)


# Helper function for plotting transfer values
def plot_transfer_values(i):
    print('Input image:')
    plt.imshow(images_test[i], interpolation='nearest')
    plt.show()

    print('Transfer values for the image using Inception model:')
    img = transfer_values_test[i]
    img = img.reshape((32, 64))
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()


plot_transfer_values(28)

# ANALYSIS OF TRANSFER VALUES USING PCA #
pca = PCA(n_components=2)
transfer_values = transfer_values_train[:3000]
cls = cls_train[:3000]

transfer_values_reduced = pca.fit_transform(transfer_values)


# Helper function for plotting the reduced transfer values:
def plot_scatter(values, cls):
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
    colors = cmap[cls]
    x = values[:, 0]
    y = values[:, 1]
    plt.scatter(x, y, color=colors)
    plt.show()


plot_scatter(transfer_values_reduced, cls)

# ANALYSIS OF TRANSFER VALUES USING T-SNE #
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)

tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)

plot_scatter(transfer_values_reduced, cls)

# NEW CLASSIFIER IN TENSORFLOW #
# Placeholder Variables
transfer_len = model.transfer_len
x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Neural Network
x_pretty = pt.wrap(x)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc_1').\
        softmax_classifier(num_classes=num_classes, labels = y_true)

# Optimization Method
global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

# Classification Accuracy
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TENSORFLOW RUN #
session = tf.Session()
session.run(tf.global_variables_initializer())

# Helper function to get a random training batch
train_batch_size = 64


def random_batch():
    num_images = len(transfer_values_train)
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch


# Helper function to perform optimization
def optimize(num_iterations):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = random_batch()
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        i_global, _ = session.run([global_step, optimizer], feed_dict=feed_dict_train)

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            batch_acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = 'Global Step: {0:>6}, Training Batch Accuracy: {1:>6.2%}'
            print(msg.format(i_global, batch_acc))

    end_time = time.time()
    time_diff = end_time - start_time
    print('Time Usage:', str(timedelta(seconds=int(round(time_diff)))))


# Helper function to plot example errors
def plot_example_errors(cls_pred, correct):
    incorrect = (correct == False)
    images = images_test[incorrect]
    cls_pred = cls_pred[incorrect]
    cls_true = cls_test[incorrect]
    n = min(16, len(images))
    plot_images(images=images[:n],
                cls_true=cls_true[:n],
                cls_pred=cls_pred[:n])


# Helper function to plot confusion matrix
def plot_confusion_matrix(cls_pred):
    cm = confusion_matrix(y_true=cls_test,
                          y_pred=cls_pred)
    for i in range(num_classes):
        class_name = '({}) {}'.format(i, class_names[i])
        print(cm[i, :], class_name)

    class_numbers = [' ({0})'.format(i) for i in range(num_classes)]
    print(''.join(class_numbers))


# Helper function for calculating classifications
batch_size = 256


def predict_cls(transfer_values, labels, cls_true):
    num_images = len(transfer_values)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred


def predict_cls_test():
    return predict_cls(transfer_values=transfer_values_test,
                       labels=labels_test,
                       cls_true=cls_test)


# Helper function for calculating the classification accuracy
def classification_accuracy(correct):
    return correct.mean(), correct.sum()


# Helper function for showing the classification accuracy
def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    correct, cls_pred = predict_cls_test()
    acc, num_correct = classification_accuracy(correct)
    num_images = len(correct)

    msg = 'Accuracy on Test Set: {0:.2%} ({1}/{2})'
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print('Example Errors:')
        plot_example_errors(cls_pred, correct)

    if show_confusion_matrix:
        print('Confusion Matrix:')
        plot_confusion_matrix(cls_pred)


# RESULTS #
optimize(2000)
print_test_accuracy(show_example_errors=True, show_confusion_matrix=True)
