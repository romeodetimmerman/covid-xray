import os
from PIL import Image
import numpy as np

RAW_DIR = "../data/raw"
INTERIM_DIR = "../data/interim"
PROCESSED_DIR = "../data/processed"
IMG_HEIGHT = IMG_WIDTH = 299
TARGET_HEIGHT = TARGET_WIDTH = 128


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
    # get base dirs
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # get subdirs
    train_covid_dir = os.path.join(train_dir, "COVID")
    train_normal_dir = os.path.join(train_dir, "NORMAL")

    val_covid_dir = os.path.join(val_dir, "COVID")
    val_normal_dir = os.path.join(val_dir, "NORMAL")

    test_covid_dir = os.path.join(test_dir, "COVID")
    test_normal_dir = os.path.join(test_dir, "NORMAL")

    # get all image paths
    train_covid_paths = [
        os.path.join(train_covid_dir, f) for f in os.listdir(train_covid_dir)
    ]
    train_normal_paths = [
        os.path.join(train_normal_dir, f) for f in os.listdir(train_normal_dir)
    ]
    val_covid_paths = [
        os.path.join(val_covid_dir, f) for f in os.listdir(val_covid_dir)
    ]
    val_normal_paths = [
        os.path.join(val_normal_dir, f) for f in os.listdir(val_normal_dir)
    ]
    test_covid_paths = [
        os.path.join(test_covid_dir, f) for f in os.listdir(test_covid_dir)
    ]
    test_normal_paths = [
        os.path.join(test_normal_dir, f) for f in os.listdir(test_normal_dir)
    ]

    # combine paths
    all_paths = (
        train_covid_paths
        + train_normal_paths
        + val_covid_paths
        + val_normal_paths
        + test_covid_paths
        + test_normal_paths
    )  # get all paths
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
    calculate mean and standard deviation of images

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
        img = np.array(Image.open(path).convert("L"))  # convert to grayscale
        all_pixels.extend(img.flatten())
        per_image_means.append(np.mean(img))
        per_image_stds.append(np.std(img))

    all_pixels = np.array(all_pixels)

    mean = all_pixels.mean()
    std = all_pixels.std()

    return mean, std


def normalize_image(input_path, output_path, mean, std):
    """
    normalize an image

    params
    ------
    input_path: str
        path to the input image
    output_path: str
        path to save the normalized image
    mean: float
        mean of pixel intensities
    std: float
        standard deviation of pixel intensities

    returns
    -------
    None
    """
    # create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    img = Image.open(input_path).convert("L")
    pixels = np.array(img.getdata())
    pixels = (pixels - mean) / std
    img.putdata(pixels.tolist())
    img.save(output_path)


def main():
    # get image paths
    all_paths = get_image_paths(RAW_DIR)
    print(f"found {len(all_paths)} images")

    # downsample images
    print(f"downsampling {len(all_paths)} images")
    interim_paths = []
    for path in all_paths:
        input_path = path
        output_path = path.replace(RAW_DIR, INTERIM_DIR)
        downsample_image(input_path, output_path, TARGET_HEIGHT, TARGET_WIDTH)
        interim_paths.append(output_path)

    # calculate mean and standard deviation of downsampled train images
    interim_train_paths = [path for path in interim_paths if "train" in path]
    print(
        f"calculating mean and standard deviation of {len(interim_train_paths)} train images"
    )
    mean, std = calc_mean_std(interim_train_paths)
    print(f"mean: {mean}, std: {std}")

    # normalize images
    print(f"normalizing {len(interim_paths)} images")
    for path in interim_paths:  # iterate over interim paths
        input_path = path  # use interim path as input
        output_path = path.replace(
            INTERIM_DIR, PROCESSED_DIR
        )  # create processed path from interim path
        normalize_image(input_path, output_path, mean, std)


if __name__ == "__main__":
    main()
