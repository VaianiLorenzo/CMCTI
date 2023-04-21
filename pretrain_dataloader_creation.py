import gc
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor
from datasets.pretrain_clip_dataset import pretrain_clip_dataset
from utils.collate_functions import pretrain_clip_collate_fn
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

    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    collate_fn = test_collate_fn = pretrain_clip_collate_fn
    

    ####################
    # TRAIN DATALOADER #
    ####################
    print("Creating train dataloader...")

    # train lists
    train_image_path = []
    train_text = []
    train_source_label = []

    for i in tqdm(range(int(len(names) * cfg.DATALOADER.PERCENTAGE_TRAIN))):
        train_image_path.append(os.path.join("data", "training", names[i]))
        train_text.append(text[i])
        train_source_label.append(source[i])
    
    train_dataloader = pretrain_clip_dataset(train_text, train_image_path, processor, train_source_label, max_length=77, isBinary = cfg.PRETRAINING.IS_BINARY)
    train_dataloader = DataLoader(train_dataloader, batch_size=cfg.DATALOADER.PRETRAINING_BATCH_SIZE, shuffle=True,
                                  num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                  prefetch_factor=4)
    
    name = "pretrain_train_clip_dataloader.bkp"
    if cfg.PRETRAINING.IS_BINARY:
        name = "pretrain_train_clip_binary_dataloader.bkp"
    torch.save(train_dataloader, os.path.join(path_output_dir, name))
    del train_dataloader
    gc.collect()

    ####################
    # VAL DATALOADER #
    ####################
    print("Creating validation dataloader...")

    # val lists
    val_image_path = []
    val_text = []
    val_source_label = []

    for i in tqdm(range(int(len(names) * cfg.DATALOADER.PERCENTAGE_TRAIN), len(names), 1)):
        val_image_path.append(os.path.join("data", "training", names[i]))
        val_text.append(text[i])
        val_source_label.append(source[i])

    val_dataloader = pretrain_clip_dataset(val_text, val_image_path, processor, val_source_label, max_length=77, isBinary = cfg.PRETRAINING.IS_BINARY)
    val_dataloader = DataLoader(val_dataloader, batch_size=cfg.DATALOADER.PRETRAINING_BATCH_SIZE, shuffle=True,
                                num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                prefetch_factor=4)
    
    name = "pretrain_val_clip_dataloader.bkp"
    if cfg.PRETRAINING.IS_BINARY:
        name = "pretrain_val_clip_binary_dataloader.bkp"
    torch.save(val_dataloader, os.path.join(path_output_dir, name))

    # ####################
    # # TEST DATALOADER #
    # ####################
    print("Creating test dataloader...")
    names, test_text, test_binary_label, test_type_label, test_source_label = read_csv_data(
            cfg.PATH.FILE_TEST_DATASET,
            cfg.MODEL.USE_REDUNDANT_LABELS,
            random_state=None)
    test_image_path = [os.path.join("data", "test", name) for name in names]

    test_dataloader = pretrain_clip_dataset(test_text, test_image_path, processor, test_source_label, max_length=77, isBinary = cfg.PRETRAINING.IS_BINARY)
    test_dataloader = DataLoader(test_dataloader, batch_size=cfg.DATALOADER.PRETRAINING_BATCH_SIZE, shuffle=True,
                                 num_workers=cfg.DATALOADER.N_WORKERS, pin_memory=True, collate_fn=test_collate_fn,
                                 prefetch_factor=4)
    
    name = "pretrain_test_clip_dataloader.bkp"
    if cfg.PRETRAINING.IS_BINARY:
        name = "pretrain_test_clip_binary_dataloader.bkp"
    torch.save(test_dataloader, os.path.join(path_output_dir, name))
