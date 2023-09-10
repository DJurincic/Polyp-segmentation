from torch.utils.data import Dataset
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

class CustomDataset(Dataset):
    """
    A standard Dataset class with the addition of allowing optional augmentations and preprocessing.

    Args:
        original_dir (str): path to folder of original images
        mask_dir (str): path to folder of mask images
        augmentation (albumentations.Compose): image augmentation pipeline
        preprocessing (albumentations.Compose): image preprocessing
    """

    def __init__(self, original_dir, mask_dir, augmentation=None, preprocessing=None):
        
        self.image_names = os.listdir(original_dir)
        self.images = [os.path.join(original_dir, image_name) for image_name in self.image_names]
        self.masks = [os.path.join(mask_dir, image_name) for image_name in self.image_names]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self.class_values=[255]

    def __getitem__(self, i):

        image = cv2.imread(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks[i], 0)

        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        if self.augmentation:
            result = self.augmentation(image=image, mask=mask)
            image, mask = result['image'], result['mask']

        if self.preprocessing:
            result = self.preprocessing(image=image, mask=mask)
            image, mask = result['image'], result['mask']

        return image, mask

    def __len__(self):

        return len(self.images)