import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as alb
import os
import random

def visualize(**images):
    """
    A function which plots all images in a single row.

    Parameters
    ----------

    images : keyword argumements in the form <image_name> = <image as numpy array>
    """

    n = len(images)
    plt.figure(figsize=(16, 5))

    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)

    plt.show()
    

def training_augmentation():
    """
    A function for augmenting training images.

    Returns:
    --------
    An albumentations.Compose object.
    """

    train_transform = [
        alb.HorizontalFlip(p=0.5),
        alb.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        alb.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        alb.RandomCrop(height=320, width=320, always_apply=True),
        alb.OneOf(
            [
                alb.RandomBrightness(p=1),
                alb.RandomGamma(p=1),
            ],
            p=1
        ),
        alb.OneOf(
            [
                alb.Blur(blur_limit=3, p=1),
                alb.MotionBlur(blur_limit=3, p=1),
            ],
            p=1
        ),
        alb.OneOf(
            [
                alb.RandomContrast(p=1),
                alb.HueSaturationValue(p=1),
            ],
            p=1
        ),
    ]
    return alb.Compose(train_transform)


def to_tensor(x, **kwargs):

    return x.transpose(2, 0, 1).astype('float32')


def validation_augmentation():
    """
    A function for augmenting validation images.

    Returns:
    --------
    An albumentations.Compose object.
    """

    transform=[
        alb.PadIfNeeded(384,480)
    ]

    return alb.Compose(transform)


def preprocessing(preprocessing_function):
    """
    A function for preprocessing images (needed for some model architectures).

    Returns:
    --------
    An albumentations.Compose object.
    """

    transform = [
        alb.Lambda(image=preprocessing_function),
        alb.Lambda(image=to_tensor, mask=to_tensor)
    ]

    return alb.Compose(transform)

def train_valid_test_split(original_dir, mask_dir, train_dir, valid_dir, test_dir, n_images, train_ratio):
    """
    A function used to separate the images from the original dataset into train, valid and test subdirectories.

    Parameters:
    -----------

    original_dir : str
            Path to the source directory of original images.
    mask_dir : str
            Path to the source directory of ground truth images.
    train_dir : str
            Path to the target directory for train images.
    valid_dir : str
            Path to the target directory for valid images.
    test_dir : str
            Path to the target directory for test images.
    n_images : int
            Number of images in the original dataset.
    train_ratio : float
            The ratio of original images used for training.
    """

    size_of_train_set = int(n_images*train_ratio)
    size_of_validation_set = int((n_images-size_of_train_set)/2)

    current_target_dir = ''

    image_names = os.listdir(original_dir)
    random.shuffle(image_names)

    for i in range(n_images):

        image = cv2.imread(f'{original_dir}/{image_names[i]}')
        mask = cv2.imread(f'{mask_dir}/{image_names[i]}')

        if i < size_of_train_set:
            current_target_dir = train_dir
        
        elif i < size_of_train_set+size_of_validation_set:
            current_target_dir = valid_dir
        
        else:
            current_target_dir = test_dir

        cv2.imwrite(f'{current_target_dir}/Original/{image_names[i]}', image)
        cv2.imwrite(f'{current_target_dir}/Ground_Truth/{image_names[i]}', mask)