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

# CREATING RANDOM TRAINING SETS
combined_images = np.concatenate([data.train.images, data.validation.images], axis=0)
combined_labels = np.concatenate([data.train.labels, data.validation.labels], axis=0)
combined_size = len(combined_images)
train_size = int(0.8 * combined_size)
validation_size = combined_size - train_size


def random_training_set():
    idx = np.random.permutation(combined_size)
    idx_train = idx[:train_size]
    idx_validation = idx[train_size:]

    x_train = combined_images[idx_train, :]
    y_train = combined_labels[idx_train, :]

    x_validation = combined_images[idx_validation, :]
    y_validation = combined_labels[idx_validation, :]

    return x_train, y_train, x_validation, y_validation


# DATA DIMENSION #
img_size = 28
img_size_flat = img_size * img_size
img_shape = (img_size, img_size)
num_channels = 1
num_classes = 10

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
saver = tf.train.Saver(max_to_keep=100)
save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)


# TENSORFLOW RUN #
# Create TensorFlow session
session = tf.Session()


# Initialize variables
def init_variables():
    session.run(tf.global_variables_initializer())


# Creating a random training batch
train_batch_size = 64


def random_batch(x_train, y_train):
    num_images = len(x_train)
    idx = np.random.choice(num_images, size=train_batch_size, replace=False)

    x_batch = x_train[idx, :]
    y_batch = y_train[idx, :]

    return x_batch, y_batch


# Optimization
def optimize(num_iterations, x_train, y_train):
    start_time = time.time()

    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            msg = 'Iter: {0:>6}, Training Batch Accuracy: {1:>6.2%}'
            print(msg.format(i + 1, acc))

    end_time = time.time()
    time_diff = end_time - start_time

    print('Time usage:', str(timedelta(seconds=int(round(time_diff)))))


# Creating ensemble of neural networks
num_networks = 5
num_iterations = 2000

if False:
    for i in range(num_networks):
        print("Neural network: {0}".format(i))
        x_train, y_train, _, _ = random_training_set()

        session.run(tf.global_variables_initializer())

        optimize(num_iterations=num_iterations,
                 x_train=x_train,
                 y_train=y_train)

        saver.save(sess=session, save_path=get_save_path(i))

        print()


# Calculating classifications
batch_size = 256


def predict_labels(images):
    num_images = len(images)

    pred_labels = np.zeros(shape=(num_images, num_classes), dtype=np.float)

    i = 0

    while i < num_images:
        j = min(i + batch_size, num_images)

        feed_dict = {x: images[i:j, :]}

        pred_labels[i:j] = session.run(y_pred, feed_dict=feed_dict)

        i = j

    return pred_labels


# Calculating and Predicting Classifications
def correct_prediction(images, labels, cls_true):
    pred_labels = predict_labels(images)
    cls_pred = np.argmax(pred_labels, axis=1)
    correct = (cls_true == cls_pred)

    return correct


def test_correct():
    return correct_prediction(images=data.test.images, labels=data.test.labels, cls_true=data.test.cls)


def validation_correct():
    return correct_prediction(images=data.validation.images, labels=data.validation.labels, cls_true=data.validation.cls)


# Calculating the classification accuracy
def classification_accuracy(correct):
    return correct.mean()


def test_accuracy():
    correct = test_correct()
    return classification_accuracy(correct)


def validation_accuracy():
    correct = validation_correct()
    return classification_accuracy(correct)


# RESULTS AND ANALYSIS
def ensemble_predictions():
    pred_labels = []
    test_accuracies = []
    val_accuracies = []

    for i in range(num_networks):
        saver.restore(sess=session, save_path=get_save_path(i))

        test_acc = test_accuracy()
        test_accuracies.append(test_acc)

        val_acc = validation_accuracy()
        val_accuracies.append(val_acc)

        msg = 'Network: {0}, Accuracy on Validation Set: {1:.4f}, Test Set: {2:.4f}'
        print(msg.format(i, val_acc, test_acc))

        pred = predict_labels(data.test.images)

        pred_labels.append(pred)

    return np.array(pred_labels), np.array(test_accuracies), np.array(val_accuracies)


pred_labels, test_accuracies, val_accuracies = ensemble_predictions()

print('Mean test-set accuracy: {0:.4f}'.format(np.mean(test_accuracies)))
print('Min test-set accuracy: {0:.4f}'.format(np.min(test_accuracies)))
print('Max test-set accuracy: {0:.4f}'.format(np.max(test_accuracies)))


# EMSEMBLE PREDICTIONS #
ensemble_pred_labels = np.mean(pred_labels, axis=0)
ensemble_cls_pred = np.argmax(ensemble_pred_labels, axis=1)

ensemble_correct = (ensemble_cls_pred == data.test.cls)
ensemble_incorrect = np.logical_not(ensemble_correct)

# BEST NEURAL NETWORK #
best_net = np.argmax(test_accuracies)
best_net_pred_labels = pred_labels[best_net, :, :]
best_net_cls_pred = np.argmax(best_net_pred_labels, axis=1)
best_net_correct = (best_net_cls_pred == data.test.cls)
best_net_incorrect = np.logical_not(best_net_correct)

# COMPARISION OF ENSEMBLE VS. THE BEST SINGLE NETWORK #
ensemble_better = np.logical_and(best_net_incorrect, ensemble_correct)
best_net_better = np.logical_and(best_net_correct, ensemble_incorrect)


# Plotting and Printing comparisons
def plot_images_comparison(idx):
    helper.plot_images(images=data.test.images[idx, :],
                       cls_true=data.test.cls[idx],
                       ensemble_cls_pred=ensemble_cls_pred[idx],
                       best_cls_pred=best_net_cls_pred[idx])


def print_labels(labels, idx, num=1):
    labels = labels[idx, :]
    labels = labels[:num, :]
    labels_rounded = np.round(labels, 2)
    print(labels_rounded)


def print_labels_ensemble(idx, **kwargs):
    print_labels(labels=ensemble_pred_labels, idx=idx, **kwargs)


def print_labels_best_net(idx, **kwargs):
    print_labels(labels=best_net_pred_labels, idx=idx, **kwargs)


def print_labels_all_nets(idx):
    for i in range(num_networks):
        print_labels(labels=pred_labels[i, :, :], idx =idx, num=1)


# Ensemble is better than the best network
plot_images_comparison(idx=ensemble_better)
print('Ensemble is better:', ensemble_better.sum())
# print_labels_ensemble(idx=ensemble_better, num=1)
# print_labels_best_net(idx=ensemble_better, num=1)
# print_labels_all_nets(idx=ensemble_better)

# Best network is better than ensemble
plot_images_comparison(idx=best_net_better)
print('Best network is better:', best_net_better.sum())
# print_labels_ensemble(idx=best_net_better, num=1)
# print_labels_best_net(idx=best_net_better, num=1)
# print_labels_all_nets(idx=best_net_better)

session.close()
