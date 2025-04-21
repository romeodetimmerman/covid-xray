import os
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
IMG_HEIGHT = IMG_WIDTH = 299
TARGET_HEIGHT = TARGET_WIDTH = 224
BATCH_SIZE = 128


def get_image_paths(data_dir):
    """
    get image paths for a given data directory

    params
    ------
    data_dir: str
        path to the data directory

    returns
    -------
    all_paths: list
        list of paths to all images
    """
    all_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".png")):
                all_paths.append(os.path.join(root, file))
    return all_paths


def downsample_image(input_path, output_path, target_height, target_width):
    """
    downsample a single image to specified height and width

    params
    ------
    input_path: str
        path to the input image
    output_path: str
        path to save the downsampled image
    target_height: int
        target height for downsampling
    target_width: int
        target width for downsampling

    returns
    -------
    None
    """
    # create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # open image and convert to grayscale
    img = Image.open(input_path).convert("L")

    # resize image
    target_size = (target_width, target_height)
    img_resized = img.resize(target_size, Image.LANCZOS)

    # save to output path
    img_resized.save(output_path)


def calc_mean_std(train_paths):
    """
    calculate mean and standard deviation of train images

    params
    ------
    input_path: str
        path to input directory

    returns
    -------
    mean: float
        mean of pixel intensities
    std: float
        standard deviation of pixel intensities
    """
    all_pixels = []
    per_image_means = []
    per_image_stds = []

    for path in train_paths:
        img = np.array(Image.open(path).convert("L"))
        all_pixels.extend(img.flatten())
        per_image_means.append(np.mean(img))
        per_image_stds.append(np.std(img))

    all_pixels = np.array(all_pixels)

    mean = all_pixels.mean()
    std = all_pixels.std()

    return mean, std


def load_data(train_dir, val_dir, test_dir, mean, std):
    """
    create data generators for train, validation, and test sets

    params
    ------
    train_dir: str
        path to training data directory
    val_dir: str
        path to validation data directory
    test_dir: str
        path to test data directory
    mean: np.ndarray
        mean pixel values (rgb) for normalization
    std: np.ndarray
        std pixel values (rgb) for normalization

    returns
    -------
    train_data_gen: DirectoryIterator
        generator for training data with augmentation
    val_data_gen: DirectoryIterator
        generator for validation data without augmentation
    test_data_gen: DirectoryIterator
        generator for test data without augmentation
    test_data_gen_raw: DirectoryIterator
        generator for test data without augmentation or normalization (for plotting)
    """

    # shared preprocessing function
    def preprocess_norm(img):
        return (img - mean) / std

    # generator for training data (with augmentation and normalization)
    train_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        preprocessing_function=preprocess_norm,
    )

    # generator for validation data (only normalization)
    val_gen = ImageDataGenerator(preprocessing_function=preprocess_norm)

    # generator for test data (only normalization)
    test_gen = ImageDataGenerator(preprocessing_function=preprocess_norm)

    # generator for raw test data (no normalization or augmentation)
    test_gen_raw = ImageDataGenerator()

    target_size = (TARGET_HEIGHT, TARGET_WIDTH)
    color_mode = "rgb"

    print("creating train generator")
    train_data_gen = train_gen.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=train_dir,
        shuffle=True,
        target_size=target_size,
        class_mode="binary",
        color_mode=color_mode,
    )

    print("creating validation generator")
    val_data_gen = val_gen.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=val_dir,
        shuffle=False,  # no shuffle for validation
        target_size=target_size,
        class_mode="binary",
        color_mode=color_mode,
    )

    print("creating test generator (normalized)")
    test_data_gen = test_gen.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=test_dir,
        shuffle=False,  # no shuffle for test
        target_size=target_size,
        class_mode="binary",
        color_mode=color_mode,
    )

    print("creating test generator (raw)")
    test_data_gen_raw = test_gen_raw.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=test_dir,
        shuffle=False,  # no shuffle for test
        target_size=target_size,
        class_mode="binary",
        color_mode=color_mode,
    )

    return train_data_gen, val_data_gen, test_data_gen, test_data_gen_raw


def main():
    # get image paths
    all_paths = get_image_paths(RAW_DIR)
    print(f"found {len(all_paths)} images")

    # downsample images
    print(f"downsampling {len(all_paths)} images to {PROCESSED_DIR}")
    processed_paths = []
    for path in all_paths:
        input_path = path

        output_path = path.replace(RAW_DIR, PROCESSED_DIR)
        downsample_image(input_path, output_path, TARGET_HEIGHT, TARGET_WIDTH)
        processed_paths.append(output_path)

    # calculate mean and std
    mean, std = calc_mean_std(processed_paths)

    # load data
    train_gen, val_gen, test_gen, test_gen_raw = load_data(
        PROCESSED_DIR, PROCESSED_DIR, PROCESSED_DIR, mean, std
    )

    return train_gen, val_gen, test_gen, test_gen_raw


if __name__ == "__main__":
    main()
