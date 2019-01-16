
# =====================================================================
#  CNN_Preprocessing.py
# =====================================================================
#
#  Author:         (c) 2019 Antonio Pio Ricciardi & Matteo Prata
#  Created:        January  02, 2019

import csv
import cv2, os
import numpy as np
from random import shuffle

DATA_PATH = "training_data/csv_file/train_data.csv"
IMAGE_DIR_PATH = "training_data/pictures"

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3

BATCH_SIZE = 50
EPOCHS = 200


def augment_turning_instances(dataset, threshold):
    """
    Given a dataset, it returns all the examples having steering angle grater than 'threshold', used to augment examples
    where the car is steering in a curve, to avoid the bias of straight lines following.

    :param dataset: the dataset to augment
    :param threshold: the value must be between 0 and 1, it's the amount of steering to accept
    :return: the all the examples having steering angle grater than 'threshold'
    """
    augmenting = []

    for inst in dataset:
        if abs(inst[3]) > threshold:
            augmenting.append(inst)

    return augmenting


def load_data(path, train_split=80):
    """
    Loads the dataset and splits it into traingset and testset coording to 'train_split'. The sataset must have the
    following format (time_stemp, mid_camera_picture_name, right_camera_picture_name, left_camera_picture_name, steering_angle).

    :param path: the path of the dataset to load in memory
    :param train_split: the percentage of data to assign to the training set. 80% to training set and 20% to test set by default
    :return: the training set and test set
    """

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        # csv file structured as: index, path1, path2, path3, class
        dataset = [(da[1],da[2],da[3],float(da[4])) for da in csv_reader]

        shuffle(dataset)
        train_data = dataset[:int(train_split*len(dataset)/100)]
        test_data = dataset[int(train_split*len(dataset)/100):]

        augenenting = augment_turning_instances(train_data, threshold=0.4)
        train_data += augenenting
        train_data += augenenting
        shuffle(dataset)

        return train_data, test_data


def show_image(image):
    """
    It shows an image on screen, the image must be a numpy array.
    """
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_image(img_dir, image_name):
    """
    It load the image in memory as a 3D numpyarray.
    """
    image_tail = os.path.join("_".join(image_name.split("_")[:2]), image_name)
    image = cv2.imread(os.path.join(img_dir, image_tail.strip()))
    return image


def randomly_chose_image(image_paths, steering):
    """
    Choose the image randomly between mid, right, left camera; adjust the steering angle accordingly.
    """
    rand_i = np.random.randint(3)
    image_path = image_paths[rand_i]

    # if image from right and left cameras are chosen,
    # adjust the steering angle
    if rand_i == 1:
        steering += 0.3

    elif rand_i == 2:
        steering -= 0.3

    return image_path, steering


def flip_image_horizontally(image, steering):
    """
    Flips the input image horizontally and adjusts the steering angle accordingly.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering = -steering

    return image, steering


def slide_image(image, steering, range_x=100, range_y=10):
    """
    Randomly shift the image vertically and horizontally, and adjusts the steering angle accordingly.
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering


def preprocess_input(image):
    """
    Apply this pre-processing to all the images passed to the CNN. Crops the sky and useless street details. Applies
    RGB-to-YUV filter.
    """
    image = image[60:-25, :, :]   # image[96:, :, :]                        # remove the sky
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)  # interpolation to a new size
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)                          # YUV filter
    return image


def augement_data(images, steering):
    """
    Augemnt the data by choosing a random image from the three views and applying a flip and a sliding.
    """
    image_path, steering = randomly_chose_image(images, steering)
    raw_image = load_image(IMAGE_DIR_PATH, image_path)

    image, steering = flip_image_horizontally(raw_image, steering)
    image, steering = slide_image(image, steering)

    return image, steering


def generate_train_batch(dataset, batch_size):
    """
    Generates training batches.
    """
    batch_images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    batch_steers = np.empty(batch_size)

    while True:
        i = 0
        for index in np.random.permutation(len(dataset)):
            x0, x1, x2, y = dataset[index]

            # With high probability chose one among mid-right-left view
            if np.random.rand() < 0.6:
                image_raw, steering = augement_data((x0, x1, x2), y)
            else:
                image_path, steering = (x0, y)
                image_raw = load_image(IMAGE_DIR_PATH, image_path)

            image_pp = preprocess_input(image_raw)

            batch_images[i] = image_pp
            batch_steers[i] = steering

            i += 1
            if i == batch_size:
                break

        yield batch_images, batch_steers


def generate_test_batch(dataset, batch_size):
    """
    Generates test batches.
    """
    batch_images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    batch_steers = np.empty(batch_size)
    d_size = 0  # how much of the dataset has been covered

    while True:
        i = 0
        for image_path, _, _, steering in dataset[d_size:d_size+batch_size]:  # window shifting at each call

            image_raw = load_image(IMAGE_DIR_PATH, image_path)
            image = preprocess_input(image_raw)

            batch_images[i] = image
            batch_steers[i] = steering

            i += 1
            if i == batch_size:
                break

        d_size += batch_size
        yield batch_images, batch_steers


