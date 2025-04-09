import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)


def check_image_size(image_paths):
    """
    check the size of the images in the list

    params
    ------
    image_paths: list
        list of paths to images

    returns
    -------
    None
    """
    if image_paths:
        # get reference size from first image
        reference_img = Image.open(image_paths[0])
        reference_size = reference_img.size
        reference_img.close()

        different_sizes = []
        for img_path in image_paths[1:]:
            img = Image.open(img_path)
            if img.size != reference_size:
                different_sizes.append((img_path, img.size))
            img.close()
            if different_sizes:
                break

        print(f"reference image size: {reference_size}")
        if different_sizes:
            print("images have different sizes")
            print(f"example of different size: {different_sizes[0]}")
        else:
            print("all images have the same size")
            print(f"(checked all {len(image_paths)} images)")
    else:
        print("no images found")


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def calculate_pixel_stats(image_paths):
    """
    calculate pixel intensity statistics for a set of images

    params
    ------
    image_paths: list
        list of paths to images

    returns
    -------
    stats: dict
        dictionary with pixel statistics
    """
    # get pixel values from all images
    all_pixels = []
    per_image_means = []
    per_image_stds = []

    for path in image_paths:
        img = np.array(Image.open(path).convert("L"))  # convert to grayscale
        all_pixels.extend(img.flatten())
        per_image_means.append(np.mean(img))
        per_image_stds.append(np.std(img))

    all_pixels = np.array(all_pixels)

    # calculate statistics
    stats = {
        "global_mean": np.mean(all_pixels),
        "global_std": np.std(all_pixels),
        "global_min": np.min(all_pixels),
        "global_max": np.max(all_pixels),
        "per_image_mean_avg": np.mean(per_image_means),
        "per_image_mean_std": np.std(per_image_means),
        "per_image_std_avg": np.mean(per_image_stds),
        "per_image_std_std": np.std(per_image_stds),
    }

    return stats


def visualize_augmentations(img_array, img_height=128, img_width=128):
    """
    visualize the augmentations of an image

    params
    ------
    img_array: numpy array
        the image to augment
    img_height: int
        the height of the image
    img_width: int
        the width of the image

    returns
    -------
    None
    """

    # ensure the image has 3 dimensions (height, width, channels)
    img_array = np.expand_dims(img_array, axis=-1)

    # add batch dimension (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # define individual augmentation generators
    gens = {
        "rotation": ImageDataGenerator(rotation_range=10),
        "width_shift": ImageDataGenerator(width_shift_range=0.05),
        "height_shift": ImageDataGenerator(height_shift_range=0.05),
        "zoom": ImageDataGenerator(zoom_range=0.1),
        "combined": ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.2,
        ),
    }

    augmented_imgs = []
    titles = []

    for name, gen in gens.items():
        aug_img = next(gen.flow(img_array, batch_size=1))[0]
        augmented_imgs.append(aug_img)
        titles.append(name)

    # plot original + augmented
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(img_array[0].squeeze(), cmap="gray")
    plt.title("original")
    plt.axis("off")

    for i, (img, title) in enumerate(zip(augmented_imgs, titles)):
        plt.subplot(2, 3, i + 2)
        plt.imshow(img.squeeze(), cmap="gray")
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()
