import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.examples.tutorials.mnist import input_data

# Load Data
data = input_data.read_data_sets('data/MNIST', one_hot=True)
print('Size of:')
print('Training set: \t\t{}'.format(len(data.train.labels)))
print('Validation set: \t\t{}'.format(len(data.validation.labels)))
print('Test set: \t\t{}'.format(len(data.test.labels)))
print(data.test.labels[0:10, :])
data.test.cls = np.array([label.argmax() for label in data.test.labels])
print(data.test.cls[0:10])

# Data Dimensions
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_classes = 10


# Helper function for plotting images
def plot_images(images, cls_true, cls_pred=None):
    assert len(images) == len(cls_true) == 9

    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image
        ax.imshow(images[i].reshape(img_shape), cmap='binary')

        # Show true and predicted classes
        if cls_pred is None:
            xlabel = 'True: {0}'.format(cls_true[i])
        else:
            xlabel = 'True: {0}, Pred: {1}'.format(cls_true[i], cls_pred[i])

        ax.set_xlabel(xlabel)

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# Sample plots
images = data.test.images[0:9]
cls_true = data.test.cls[0:9]
plot_images(images, cls_true)

# TensorFlow Graph

# Placeholder Variables
x = tf.placeholder(tf.float32, [None, img_size_flat])
y_true = tf.placeholder(tf.float32, [None, num_classes])
y_true_sparse = tf.argmax(y_true, axis=1)
y_true_cls = tf.placeholder(tf.int64, [None])

# Variables to be optimized
weights = tf.Variable(tf.zeros([img_size_flat, num_classes]))
biases = tf.Variable(tf.zeros([num_classes]))

# Model
logits = tf.matmul(x, weights) + biases
y_pred = tf.nn.softmax(logits)
y_pred_cls = tf.argmax(y_pred, axis=1)

# Cost function to be optimized
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_true_sparse)
cost = tf.reduce_mean(cross_entropy)

# Optimization Method *Change the learning rate and optimizer*
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(cost)

# Performance Measure
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TensorFlow Run
session = tf.Session()
session.run(tf.global_variables_initializer())

# Helper function to perform optimization iterations *Change the batch size*
batch_size = 1000


def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)


# Helper function to show performance
feed_dict_test = {x: data.test.images, y_true: data.test.labels, y_true_cls: data.test.cls}


def print_accuracy():
    acc = session.run(accuracy, feed_dict=feed_dict_test)
    print('Accuracy on test-set: {0:.1%}'.format(acc))


def print_confusion_matrix():
    cls_true = data.test.cls
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
    print(cm)

    plt.imshow(cm, interpolation='nearest')
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_example_errors():
    correct, cls_pred, out, pred = session.run([correct_prediction, y_pred_cls, logits, y_pred], feed_dict=feed_dict_test)

    incorrect = (correct == False)

    images = data.test.images[incorrect]

    cls_pred = cls_pred[incorrect]
    cls_true = data.test.cls[incorrect]

    out = out[incorrect]
    pred = pred[incorrect]

    plot_images(images[0:9], cls_true[0:9], cls_pred[0:9])

    print(out[0:9])
    print(pred[0:9])


# Helper function to plot the model weights
def plot_weights():
    w = session.run(weights)
    w_min = np.min(w)
    w_max = np.max(w)

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.4)

    for i, ax in enumerate(axes.flat):
        if i < 10:
            image = w[:, i].reshape(img_shape)
            ax.set_xlabel('Weights: {}'.format(i))
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# Performance after 2000 optimization iteration
optimize(num_iterations=2000)
print_accuracy()
plot_example_errors()
plot_weights()
print_confusion_matrix()

# Close
session.close()