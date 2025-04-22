import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from keras.preprocessing.image import array_to_img, img_to_array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    generate gradcam heatmap for model visualization

    params
    ------
    img_array: numpy.ndarray
        preprocessed image array
    model: tf.keras.Model
        tensorflow keras model
    last_conv_layer_name: str
        name of last conv layer
    pred_index: int, optional
        index of predicted class

    returns
    -------
    numpy.ndarray
        normalized gradcam heatmap
    """
    print(f"input array shape: {img_array.shape}")

    # get last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)

    # find layer index
    last_conv_idx = None
    for idx, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            last_conv_idx = idx
            break

    if last_conv_idx is None:
        raise ValueError(f"Could not find layer {last_conv_layer_name}")

    # split model at conv layer
    conv_model = tf.keras.Sequential(model.layers[: last_conv_idx + 1])
    rest_model = tf.keras.Sequential(model.layers[last_conv_idx + 1 :])

    # compute gradients
    with tf.GradientTape() as tape:
        conv_output = conv_model(img_array)
        tape.watch(conv_output)
        predictions = rest_model(conv_output)
        pred_index = tf.argmax(predictions[0]) if pred_index is None else pred_index
        class_channel = predictions[:, pred_index]

    # calculate heatmap
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # normalize to 0-1
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / tf.reduce_max(heatmap)
    return heatmap.numpy()


def load_images(image_paths, size):
    """
    load and preprocess batch of images

    params
    ------
    image_paths: list
        paths to images
    size: tuple
        target image size (height, width)

    returns
    -------
    tuple
        processed image batch and display image batch
    """
    image_list = []
    display_images = []

    for image_path in image_paths:
        # load and decode
        img = tf.image.decode_image(tf.io.read_file(image_path))

        # convert grayscale to rgb if needed
        if img.shape[-1] == 1:
            img = tf.image.grayscale_to_rgb(img)

        # resize
        img_resized = tf.image.resize(img, size)
        display_images.append(img_resized)

        # preprocess for model
        img_processed = tf.cast(img_resized, tf.float32) / 127.5 - 1
        image_list.append(img_processed)

    return tf.stack(image_list), tf.stack(display_images)


def display_heatmap(img, heatmap, alpha=0.4):
    """
    display heatmap overlaid on image

    params
    ------
    img: numpy.ndarray
        original image
    heatmap: numpy.ndarray
        attribution heatmap
    alpha: float
        heatmap transparency
    """
    # prep image
    img = img[0] if img.ndim == 4 else img
    img = tf.cast(img, tf.uint8).numpy() if img.dtype != np.uint8 else img

    # prep heatmap
    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = np.uint8(255 * heatmap)

    # colorize heatmap
    jet_colors = mpl.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # resize and blend
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # create final image
    result = jet_heatmap * alpha + img * (1 - alpha)
    result = np.clip(result, 0, 255).astype(np.uint8)

    # display
    plt.figure(figsize=(10, 10))
    plt.imshow(result)
    plt.axis("off")
    plt.show()
