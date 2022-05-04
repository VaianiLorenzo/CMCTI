import gc
from operator import index
import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets.MAMI_binary_dataset import MAMI_binary_dataset
from datasets.MAMI_test_binary_dataset import MAMI_test_binary_dataset
from datasets.MAMI_test_vb_binary_dataset import MAMI_test_vb_binary_dataset
from datasets.MAMI_vb_binary_dataset import MAMI_vb_binary_dataset
from datasets.MAMI_vb_multitask_dataset import MAMI_vb_multitask_dataset
from utils.collate_functions import base_collate_fn, base_test_collate_fn, vb_collate_fn, vb_test_collate_fn, \
    multitask_collate_fn
from utils.utils import read_config, read_csv_data

path_output_dir = os.path.join("data", "dataloaders")

# Always use the same train/validation split
random_state = 1995

#def compute_weights(labels):



if __name__ == "__main__":
    cfg = read_config()

    # Create output directory if does not exist
    if not os.path.isdir(path_output_dir):
        os.mkdir(path_output_dir)

    names, text, misogynous, type_label, source = read_csv_data(cfg.PATH.FILE_TRAIN_DATASET, random_state=random_state)

    if cfg.MODEL.TYPE == "base":
        text_model_name = "bert-base-cased"
        collate_fn = base_collate_fn
        test_collate_fn = base_test_collate_fn
    elif cfg.MODEL.TYPE == "visual_bert":
        text_model_name = "bert-base-uncased"
        collate_fn = vb_collate_fn
        test_collate_fn = vb_test_collate_fn
    elif cfg.MODEL.TYPE == "multitask":
        text_model_name = "bert-base-uncased"
        collate_fn = test_collate_fn = multitask_collate_fn

    # BERT tokenizer for text
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    ####################
    # TRAIN DATALOADER #
    ####################
    print("Creating train dataloader...")

    # train lists
    train_image_path = []
    train_text = []
    train_binary_label = []
    train_type_label = []
    train_source_label = []

    for i in tqdm(range(int(len(names) * cfg.DATALOADER.PERCENTAGE_TRAIN))):
        train_image_path.append(os.path.join("data", "training", names[i]))
        train_text.append(text[i])
        train_binary_label.append(misogynous[i])
        train_type_label.append(type_label[i])
        train_source_label.append(source[i])

    '''
    # SOURCE WEIGHTS
    source_count = np.array(sum(train_source_label))
    source_weights = 1. / source_count
    source_samples_weights = []
    for l in train_source_label:
        indexes = np.where(l == 1)[0]
        weight = 0
        for idx in indexes:
            weight = weight + source_weights[idx]
        weight = weight / len(indexes)
        source_samples_weights.append(weight)

    # TYPE WEIGHTS
    type_count = list(sum(train_type_label))
    count = 0
    for a in train_type_label:
        if list(a) == [0,0,0,0,0]:
            count += 1
    type_count.insert( 0, count)
    type_count = np.asarray(type_count)
    type_weights = 1. / type_count
    type_samples_weights = []
    for l in train_type_label:
        indexes = np.where(l == 1)[0]
        weight = 0
        if len(indexes) == 0:
            weight = type_weights[0]
        else:
            for idx in indexes:
                weight = weight + type_weights[idx+1]
            weight = weight / len(indexes)
        type_samples_weights.append(weight)

    weights = np.asarray(source_samples_weights) + np.asarray(type_samples_weights)
    '''

    
    source_count = np.array(sum(train_source_label))
    source_weights = 1. / source_count
    source_weights = source_weights / sum(source_weights)
    np.save("data/source_weights.npy", source_weights)
        
    type_count = list(sum(train_type_label))
    type_count = np.asarray(type_count)
    type_weights = 1. - type_count / len(train_type_label)
    np.save("data/type_weights.npy", type_weights)


    if cfg.MODEL.TYPE == "base":
        train_dataloader = MAMI_binary_dataset(train_text, train_image_path, text_tokenizer, train_binary_label,
                                               max_length=128)
    elif cfg.MODEL.TYPE == "visual_bert":
        train_dataloader = MAMI_vb_binary_dataset(train_text, train_image_path, text_tokenizer, train_binary_label,
                                                  max_length=128)
    elif cfg.MODEL.TYPE == "multitask":
        train_dataloader = MAMI_vb_multitask_dataset(train_text, train_image_path, text_tokenizer, train_binary_label,
                                                     train_type_label, train_source_label, max_length=128)

    train_dataloader = DataLoader(train_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                  prefetch_factor=4)
    torch.save(train_dataloader, os.path.join(path_output_dir, f"train_{cfg.MODEL.TYPE}_dataloader.bkp"))
    del train_dataloader
    gc.collect()

    ####################
    # VAL DATALOADER #
    ####################
    print("Creating validation dataloader...")

    # val lists
    val_image_path = []
    val_text = []
    val_binary_label = []
    val_type_label = []
    val_source_label = []

    for i in tqdm(range(int(len(names) * cfg.DATALOADER.PERCENTAGE_TRAIN), len(names), 1)):
        val_image_path.append(os.path.join("data", "training", names[i]))
        val_text.append(text[i])
        val_binary_label.append(misogynous[i])
        val_type_label.append(type_label[i])
        val_source_label.append(source[i])

    if cfg.MODEL.TYPE == "base":
        val_dataloader = MAMI_binary_dataset(val_text, val_image_path, text_tokenizer, val_binary_label,
                                             max_length=128)
    elif cfg.MODEL.TYPE == "visual_bert":
        val_dataloader = MAMI_vb_binary_dataset(val_text, val_image_path, text_tokenizer, val_binary_label,
                                                max_length=128)
    elif cfg.MODEL.TYPE == "multitask":
        val_dataloader = MAMI_vb_multitask_dataset(val_text, val_image_path, text_tokenizer, val_binary_label,
                                                   val_type_label, val_source_label, max_length=128)
    val_dataloader = DataLoader(val_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                prefetch_factor=4)
    torch.save(val_dataloader, os.path.join(path_output_dir, f"val_{cfg.MODEL.TYPE}_dataloader.bkp"))
'''
    ####################
    # TEST DATALOADER #
    ####################
    print("Creating test dataloader...")

    if cfg.MODEL.TYPE == "base" or cfg.MODEL.TYPE == "visual_bert":
        df = pd.read_csv(cfg.PATH.FILE_TEST_DATASET, sep="\t")
        images = list(df["file_name"])
        for i in range(len(images)):
            images[i] = os.path.join("data", "test", images[i])
        texts = list(df["Text Transcription"])

        if cfg.MODEL.TYPE == "base":
            test_dataloader = MAMI_test_binary_dataset(texts, images, text_tokenizer, max_length=128)
        elif cfg.MODEL.TYPE == "visual_bert":
            test_dataloader = MAMI_test_vb_binary_dataset(texts, images, text_tokenizer, max_length=128)
    elif cfg.MODEL.TYPE == "multitask":
        names, test_text, test_binary_label, test_type_label, test_source_label = read_csv_data(cfg.PATH.FILE_TEST_DATASET,
                                                                               random_state=None)
        test_image_path = [os.path.join("data", "test", name) for name in names]
        test_dataloader = MAMI_vb_multitask_dataset(test_text, val_image_path, text_tokenizer, test_binary_label,
                                                    test_type_label, test_source_label, max_length=128)

    test_dataloader = DataLoader(test_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                 num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=test_collate_fn,
                                 prefetch_factor=4)
    torch.save(test_dataloader, os.path.join(path_output_dir, f"test_{cfg.MODEL.TYPE}_dataloader.bkp"))
'''