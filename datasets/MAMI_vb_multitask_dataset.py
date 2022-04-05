import torch.utils.data as data
from PIL import Image
import numpy as np


class MAMI_vb_multitask_dataset(data.Dataset):

    def __init__(self, text, image_path, text_processor, binary_label, type_label, source_modality_label,
                 max_length=128):
        self.text_processor = text_processor
        self.text = text
        self.binary_label = binary_label
        self.type_label = type_label
        self.source_modality_label = source_modality_label
        self.max_length = max_length
        self.image_path = image_path

    def __getitem__(self, index):
        return self.text_processor(self.text[index], padding="max_length", max_length=self.max_length, truncation=True,
                                   return_tensors='pt'), self.image_path[index], self.binary_label[index], \
               self.type_label[index], self.source_modality_label[index]

    def load_image(self, filename):
        img = Image.open(filename)
        img.load()
        data = np.asarray(img, dtype="float32")
        return data

    def __len__(self):
        return len(self.text)
