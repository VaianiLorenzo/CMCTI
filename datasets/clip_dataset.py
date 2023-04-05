from os import truncate
from unittest.util import _MAX_LENGTH
import torch.utils.data as data
from PIL import Image
import numpy as np


class clip_dataset(data.Dataset):

    def __init__(self, text, image, processor, binary_label, type_label, source_modality_label,
                 max_length=77, augmentation=False):
        self.processor = processor
        self.text = text
        self.binary_label = binary_label
        self.type_label = type_label
        self.source_modality_label = source_modality_label
        self.max_length = max_length
        self.image = image
        self.augmentation = augmentation

        if self.augmentation:
            self.mysogyny_texts = []
            for i in range(len(self.text)):
                if np.argmax(self.source_modality_label[i]) == 1 or np.argmax(self.source_modality_label[i]) == 3:
                    self.mysogyny_texts.append(self.text[i])
            self.mysogyny_images = []
            for i in range(len(self.image)):
                if np.argmax(self.source_modality_label[i]) == 2 or np.argmax(self.source_modality_label[i]) == 3:
                    self.mysogyny_images.append(self.image[i])
                    
    def __getitem__(self, index):
        sml = self.source_modality_label[index]
        if self.augmentation:
            if np.random.choice([0,1], p=[0.5, 0.5]):
                if np.argmax(sml) == 1:
                    idx = np.random.randint(0, len(self.mysogyny_texts))
                    return self.processor(text=[self.mysogyny_texts[idx]], images=[Image.open(self.image[index])], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True), \
                        self.binary_label[index], self.type_label[index], self.source_modality_label[index]
                elif np.argmax(sml) == 2:
                    idx = np.random.randint(0, len(self.mysogyny_images))
                    return self.processor(text=[self.text[index]], images=[Image.open(self.mysogyny_images[idx])], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True), \
                        self.binary_label[index], self.type_label[index], self.source_modality_label[index]
                elif np.argmax(sml) == 3:
                    idx = np.random.randint(0, len(self.mysogyny_texts))
                    idy = np.random.randint(0, len(self.mysogyny_images))
                    return self.processor(text=[self.mysogyny_texts[idx]], images=[Image.open(self.mysogyny_images[idy])], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True), \
                        self.binary_label[index], self.type_label[index], self.source_modality_label[index]

        return self.processor(text=[self.text[index]], images=[Image.open(self.image[index])], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True), \
            self.binary_label[index], self.type_label[index], self.source_modality_label[index]

    def __len__(self):
        return len(self.text)