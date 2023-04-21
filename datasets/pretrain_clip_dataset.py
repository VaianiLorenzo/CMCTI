import torch
import torch.utils.data as data
from PIL import Image
import numpy as np


class pretrain_clip_dataset(data.Dataset):

    def __init__(self, text, image, processor, source_modality_label, max_length=77, isBinary = False):
        self.processor = processor
        self.text = text
        self.source_modality_label = source_modality_label
        self.max_length = max_length
        self.image = image
        self.isBinary = isBinary

    def __getitem__(self, index):

        isPositive = np.random.choice([-1, 1], p=[0.5, 0.5])
        first_label = self.source_modality_label[index]
        first_label = np.argmax(first_label)
        if self.isBinary:
            first_label = 1 if first_label == 4 else 0

        second_label = -1
        
        while second_label == -1 or (isPositive==1 and first_label != second_label) or (isPositive==-1 and first_label == second_label):
            second_index = np.random.randint(0, len(self.text))
            second_label = np.argmax(self.source_modality_label[second_index])
            if self.isBinary:
                second_label = 1 if second_label == 4 else 0

        print("isPositive:", isPositive, " - first_label:", first_label, " - second_label:", second_label)

        return self.processor(text=[self.text[index]], images=[Image.open(self.image[index])], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True), \
            self.processor(text=[self.text[second_index]], images=[Image.open(self.image[second_index])], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True), \
            isPositive

    def __len__(self):
        return len(self.text)