import numpy as np
import tensorflow as tf
import time
from datetime import timedelta
import os
import prettytensor as pt
import helper

# LOAD DATA #
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)
data.validation.cls = np.argmax(data.validation.labels, axis=1)

# DATA DIMENSION #
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

helper.plot_images(data.test.images[:16], 4, img_shape, data.test.cls[:16])

# TENSORFLOW GRAPH #
# Placeholder Variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

# Neural Network
x_pretty = pt.wrap(x_image)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        conv2d(kernel=5, depth=16, name='layer_conv_1').\
        max_pool(kernel=2, stride=2).\
        conv2d(kernel=5, depth=36, name='layer_conv_2').\
        max_pool(kernel=2, stride=2).\
        flatten().\
        fully_connected(size=128, name='layer_fc_1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)


# Getting the weights
def get_weights_variable(layer_name):
    with tf.variable_scope(layer_name, reuse=True):
        variable = tf.get_variable('weights')

    return variable


weights_conv_1 = get_weights_variable(layer_name='layer_conv_1')
weights_conv_2 = get_weights_variable(layer_name='layer_conv_2')

# Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

# Performance Measures
y_pred_cls = tf.argmax(y_pred, axis=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Saver
saver = tf.train.Saver()
save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'best_validation')

# TENSORFLOW RUN #
# Create TensorFlow session
session = tf.Session()


# Initialize variables
def init_variables():
    session.run(tf.global_variables_initializer())


init_variables()

# Optimization
train_batch_size = 64
best_validation_accuracy = 0.0
last_improvement = 0
require_improvement = 1000

total_iterations = 0


def optimize(num_iterations):
    global total_iterations
    global best_validation_accuracy
    global last_improvement

    start_time = time.time()

    for i in range(num_iterations):
        total_iterations += 1

        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if (total_iterations % 100 == 0) or (i == num_iterations - 1):
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            acc_validation, _ = validation_accuracy()

            if acc_validation > best_validation_accuracy:
                best_validation_accuracy = acc_validation
                last_improvement = total_iterations
                saver.save(sess=session, save_path=save_path)
                improved_str = '*'
            else:
                improved_str = ''

            msg = 'Iter: {0:>6}, Train-Batch Acc: {1:>6.2%}, Validation Acc: {2:>6.2%} {3}'
            print(msg.format(i + 1, acc_train, acc_validation, improved_str))

        if total_iterations - last_improvement > require_improvement:
            print('No improvement found in a while, stopping optimization.')
            break

    end_time = time.time()
    time_diff = end_time - start_time

    print('Time usage:', str(timedelta(seconds=int(round(time_diff)))))


# Calculating classifications
batch_size = 256


def predict_cls(images, labels, cls_true):
    num_images = len(images)
    cls_pred = np.zeros(shape=num_images, dtype=np.int)
    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)

        feed_dict = {x: images[i:j, :], y_true: labels[i:j, :]}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i = j

    correct = (cls_true == cls_pred)

    return correct, cls_pred


def predict_cls_test():
    return predict_cls(images=data.test.images, labels=data.test.labels, cls_true=data.test.cls)


def predict_cls_validation():
    return predict_cls(images=data.validation.images, labels=data.validation.labels, cls_true=data.validation.cls)


# Classification Accuracy
def cls_accuracy(correct):
    correct_sum = correct.sum()
    acc = float(correct_sum)/len(correct)
    return acc, correct_sum


def validation_accuracy():
    correct, _ = predict_cls_validation()
    return cls_accuracy(correct)


# Showing the performance
def print_test_accuracy(show_example_errors=False, show_confusion_matrix=False):
    correct, cls_pred = predict_cls_test()
    acc, num_correct = cls_accuracy(correct)
    num_images = len(correct)

    msg = 'Accuracy on Test-set: {0:.2%} ({1}/{2})'
    print(msg.format(acc, num_correct, num_images))

    if show_example_errors:
        print('Example Errors:')
        helper.plot_example_errors(data, cls_pred, correct)

    if show_confusion_matrix:
        print('Confusion Matrix:')
        helper.plot_confusion_matrix(cls_pred, data.test.cls, num_classes)


# optimize(10000)
# print_test_accuracy()
# helper.plot_conv_weights(session, weights=weights_conv_1)
# helper.plot_conv_weights(session, weights=weights_conv_2)
#
# init_variables()
# print_test_accuracy()

# RESTORE BEST VARIABLES #
saver.restore(sess=session, save_path=save_path)
print_test_accuracy(show_example_errors=True)
