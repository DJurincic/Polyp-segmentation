import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as alb

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

    transform = [
        alb.HorizontalFlip(p=0.5),
        alb.OneOf([
            alb.RandomBrightness(p=1),
            alb.RandomGamma(p=1)
        ]),
        alb.OneOf([
            alb.Blur(blur_limit=3, p=1),
            alb.MotionBlur(blur_limit=3, p=1)
        ]),
        alb.ShiftScaleRotate(scale_limit=0.3, rotate_limit=0, p=1, border_mode=0),
        alb.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        alb.RandomCrop(height=320, width=320, always_apply=True)

    ]

    return alb.Compose(transform)


def to_tensor(x):

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