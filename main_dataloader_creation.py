import gc
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import CLIPProcessor

from datasets.baseline_dataset import baseline_dataset
from datasets.vb_dataset import vb_dataset
from datasets.clip_dataset import clip_dataset
from utils.collate_functions import baseline_collate_fn, clip_collate_fn, vb_collate_fn
from utils.utils import read_config, read_csv_data

path_output_dir = os.path.join("data", "dataloaders")

# Always use the same train/validation split
random_state = 1995

if __name__ == "__main__":
    cfg = read_config()

    # Create output directory if does not exist
    if not os.path.isdir(path_output_dir):
        os.mkdir(path_output_dir)

    names, text, misogynous, type_label, source = read_csv_data(cfg.PATH.FILE_TRAIN_DATASET,
                                                                cfg.MODEL.USE_REDUNDANT_LABELS,
                                                                random_state=random_state)

    if cfg.MODEL.TYPE == "baseline":
        text_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        collate_fn = test_collate_fn = baseline_collate_fn
    elif cfg.MODEL.TYPE == "clip-base":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        collate_fn = test_collate_fn = clip_collate_fn
    elif cfg.MODEL.TYPE == "clip-large":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        collate_fn = test_collate_fn = clip_collate_fn
    elif cfg.MODEL.TYPE == "visual_bert":
        text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        collate_fn = test_collate_fn = vb_collate_fn
    else:
        raise ValueError("Model type not supported")
    

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

    # Save file of weights for balancing
    source_count = np.array(sum(train_source_label))
    source_weights = 1. / source_count
    source_weights = source_weights / sum(source_weights)
    np.save(cfg.PATH.FILE_SOURCE_WEIGHTS, source_weights)

    type_count = list(sum(train_type_label))
    type_count = np.asarray(type_count)
    type_weights = 1. - type_count / len(train_type_label)
    np.save(cfg.PATH.FILE_TYPE_WEIGHTS, type_weights)

    if cfg.MODEL.TYPE == "baseline":
        train_dataloader = baseline_dataset(train_text, train_image_path, text_tokenizer, train_binary_label, 
                                                     train_type_label, train_source_label, max_length=128)
    elif cfg.MODEL.TYPE == "clip-base" or cfg.MODEL.TYPE == "clip-large":
        train_dataloader = clip_dataset(train_text, train_image_path, processor, train_binary_label,
                                                     train_type_label, train_source_label, max_length=77)
    elif cfg.MODEL.TYPE == "visual_bert":
        train_dataloader = vb_dataset(train_text, train_image_path, text_tokenizer, train_binary_label,
                                                     train_type_label, train_source_label, max_length=128)


    train_dataloader = DataLoader(train_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                  prefetch_factor=4)

    torch.save(train_dataloader, os.path.join(path_output_dir, f"train_{cfg.MODEL.TYPE}_dataloader.bkp"))
    del train_dataloader
    gc.collect()
    
    if cfg.MODEL.TYPE == "clip-base" or cfg.MODEL.TYPE == "clip-large":
        train_dataloader = clip_dataset(train_text, train_image_path, processor, train_binary_label,
                                                     train_type_label, train_source_label, max_length=77, augmentation=True)
        train_dataloader = DataLoader(train_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                    num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                    prefetch_factor=4)
        torch.save(train_dataloader, os.path.join(path_output_dir, f"augmentation_train_{cfg.MODEL.TYPE}_dataloader.bkp"))
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

    if cfg.MODEL.TYPE == "baseline":
        val_dataloader = baseline_dataset(val_text, val_image_path, text_tokenizer, val_binary_label,
                                                   val_type_label, val_source_label, max_length=128)
    elif cfg.MODEL.TYPE == "clip-base" or cfg.MODEL.TYPE == "clip-large":
        val_dataloader = clip_dataset(val_text, val_image_path, processor, val_binary_label,
                                                   val_type_label, val_source_label, max_length=77)
    elif cfg.MODEL.TYPE == "visual_bert":
        val_dataloader = vb_dataset(val_text, val_image_path, text_tokenizer, val_binary_label,
                                                   val_type_label, val_source_label, max_length=128)
    val_dataloader = DataLoader(val_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                prefetch_factor=4)
    torch.save(val_dataloader, os.path.join(path_output_dir, f"val_{cfg.MODEL.TYPE}_dataloader.bkp"))

    # ####################
    # # TEST DATALOADER #
    # ####################
    print("Creating test dataloader...")
    names, test_text, test_binary_label, test_type_label, test_source_label = read_csv_data(
            cfg.PATH.FILE_TEST_DATASET,
            cfg.MODEL.USE_REDUNDANT_LABELS,
            random_state=None)
    test_image_path = [os.path.join("data", "test", name) for name in names]

    if cfg.MODEL.TYPE == "baseline":
        test_dataloader = baseline_dataset(test_text, test_image_path, text_tokenizer, test_binary_label,
                                                    test_type_label, test_source_label,max_length=128)
    if cfg.MODEL.TYPE == "clip-base" or cfg.MODEL.TYPE == "clip-large":
        test_dataloader = clip_dataset(test_text, test_image_path, processor, test_binary_label,
                                                    test_type_label, test_source_label,max_length=77)
    elif cfg.MODEL.TYPE == "visual_bert":
        test_dataloader = vb_dataset(test_text, test_image_path, text_tokenizer, test_binary_label,
                                                    test_type_label, test_source_label, max_length=128)

    test_dataloader = DataLoader(test_dataloader, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                 num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=test_collate_fn,
                                 prefetch_factor=4)
    torch.save(test_dataloader, os.path.join(path_output_dir, f"test_{cfg.MODEL.TYPE}_dataloader.bkp"))
