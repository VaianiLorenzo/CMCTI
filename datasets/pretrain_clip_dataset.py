import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class pretrain_clip_dataset(data.Dataset):

    def __init__(self, text, image, processor, source_modality_label, max_length=77):
        self.processor = processor
        self.text = text
        self.source_modality_label = source_modality_label
        self.max_length = max_length
        self.image = image

    def __getitem__(self, index):

        isPositive = np.random.choice([0, 1], p=[0.5, 0.5])
        first_label = self.source_modality_label[index]
        first_label = np.argmax(first_label)
        second_label = -1
            
        while second_label == -1 or (isPositive and first_label != second_label) or (not isPositive and first_label == second_label):
            second_index = np.random.randint(0, len(self.text))
            second_label = np.argmax(self.source_modality_label[second_index])

        return self.processor(text=[self.text[index]], images=[Image.open(self.image[index])], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True), \
            self.processor(text=[self.text[second_index]], images=[Image.open(self.image[second_index])], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True), \
            isPositive

    def __len__(self):
        return len(self.text)