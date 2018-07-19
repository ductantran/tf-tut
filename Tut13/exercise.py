import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import inception

# INCEPTION MODEL #
inception.maybe_download()


# Helper function to get name of the convolutional layers
def get_conv_layer_names():
    model = inception.Inception()
    names = [op.name for op in model.graph.get_operations() if op.type == 'Conv2D']
    model.close()

    return names


conv_names = get_conv_layer_names()
print('Number of convolutional layers:', len(conv_names))


# HELPER FUNCTIONS FOR FINDING THE INPUT IMAGE #
def optimize_image(conv_id=None, feature=0, num_iterations=30, show_progress=True):
    model = inception.Inception()

    resized_image = model.resized_image

    y_pred = model.y_pred

    if conv_id is None:
        loss = model.y_logits[0, feature]
    else:
        conv_name = conv_names[conv_id]
        tensor = model.graph.get_tensor_by_name(conv_name + ':0')
        with model.graph.as_default():
            loss = tf.reduce_mean(tensor[:, :, :, feature])

    gradient = tf.gradients(loss, resized_image)

    session = tf.Session(graph=model.graph)

    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0

    for i in range(num_iterations):
        feed_dict = {model.tensor_name_resized_image: image}
        pred, grad, loss_value = session.run([y_pred, gradient, loss], feed_dict=feed_dict)
        grad = np.array(grad).squeeze()
        step_size = 1.0 / (grad.std() + 1e-8)
        image += step_size * grad
        image = np.clip(image, 0.0, 255.0)

        if show_progress:
            print('Iteration:', i)
            pred = np.squeeze(pred)
            pred_cls = np.argmax(pred)
            cls_name = model.name_lookup.cls_to_name(pred_cls, only_first_name=True)
            cls_score = pred[pred_cls]

            msg = 'Predicted class name: {0} (#{1}), score: {2:>7.2%}'
            print(msg.format(cls_name, pred_cls, cls_score))

            msg = 'Gradient min: {0:>9.6f}, max: {1:>9.6f}, step-size: {2:>9.2f}'
            print(msg.format(grad.min(), grad.max(), step_size))

            print('Loss:', loss_value)
            print()

    model.close()

    return image.squeeze()


def optimize_image_v2(conv_id=None, feature=0, num_iterations=30, show_progress=True):
    model = inception.Inception()

    resized_image = model.resized_image

    y_pred = model.y_pred

    if conv_id is None:
        loss = model.y_logits[0, feature]
    else:
        conv_name = conv_names[conv_id]
        tensor = model.graph.get_tensor_by_name(conv_name + ':0')
        with model.graph.as_default():
            loss = tf.norm(tensor[:, :, :, feature])

    gradient = tf.gradients(loss, resized_image)

    session = tf.Session(graph=model.graph)

    image_shape = resized_image.get_shape()
    image = np.random.uniform(size=image_shape) + 128.0

    for i in range(num_iterations):
        feed_dict = {model.tensor_name_resized_image: image}
        pred, grad, loss_value = session.run([y_pred, gradient, loss], feed_dict=feed_dict)
        grad = np.array(grad).squeeze()
        step_size = 1.0 / (grad.std() + 1e-8)
        image += step_size * grad
        image = np.clip(image, 0.0, 255.0)

        if show_progress:
            print('Iteration:', i)
            pred = np.squeeze(pred)
            pred_cls = np.argmax(pred)
            cls_name = model.name_lookup.cls_to_name(pred_cls, only_first_name=True)
            cls_score = pred[pred_cls]

            msg = 'Predicted class name: {0} (#{1}), score: {2:>7.2%}'
            print(msg.format(cls_name, pred_cls, cls_score))

            msg = 'Gradient min: {0:>9.6f}, max: {1:>9.6f}, step-size: {2:>9.2f}'
            print(msg.format(grad.min(), grad.max(), step_size))

            print('Loss:', loss_value)
            print()

    model.close()

    return image.squeeze()


# Helper function for plotting image and noise
def normalize_image(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / (x_max - x_min)

    return x_norm


def plot_image(image):
    img_norm = normalize_image(image)

    plt.imshow(img_norm, interpolation='nearest')
    plt.show()


def plot_images(images, show_size=100):
    fig, axes = plt.subplots(2, 3)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        img = images[i, :show_size, :show_size, :]
        img_norm = normalize_image(img)
        ax.imshow(img_norm, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()


# Helper function for optimizing and plotting images
def optimize_images(conv_id=None, num_iterations=30, show_size=100):
    if conv_id is None:
        print('Final fully-connected layer before softmax')
    else:
        print('Layer:', conv_names[conv_id])

    images = []

    for feature in range(1,7):
        print('Optimizing image for feature:', feature)
        image = optimize_image(
            conv_id=conv_id,
            feature=feature,
            show_progress=False,
            num_iterations=num_iterations
        )
        image = image.squeeze()
        images.append(image)

    images = np.array(images)
    plot_images(images=images, show_size=show_size)


# RESULTS #
# Optimize a single image for an early convolutional layer
# image = optimize_image(
#     conv_id=1,
#     feature=1,
#     num_iterations=30,
#     show_progress=True
# )
# plot_image(image)
#
# image = optimize_image_v2(
#     conv_id=1,
#     feature=1,
#     num_iterations=30,
#     show_progress=True
# )
# plot_image(image)

# Optimize multiple images for convolutional layers
# optimize_images(
#     conv_id=0,
#     num_iterations=10
# )

#
# optimize_images(
#     conv_id=5,
#     num_iterations=30
# )
#
# optimize_images(
#     conv_id=10,
#     num_iterations=30
# )
#
# optimize_images(
#     conv_id=50,
#     num_iterations=30
# )
#
# optimize_images(
#     conv_id=93,
#     num_iterations=30
# )

# Final fully connected layer before Softmax
image = optimize_image(
    conv_id=None,
    feature=1,
    num_iterations=50,
    show_progress=True
)
plot_image(image)

image = optimize_image_v2(
    conv_id=None,
    feature=1,
    num_iterations=50,
    show_progress=True
)
plot_image(image)
