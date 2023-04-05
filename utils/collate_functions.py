import torch

def baseline_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    label_binary = [item[2] for item in batch]
    label_type = [item[3] for item in batch]
    label_source = [item[4] for item in batch]
    return [text, img, label_binary, label_type, label_source]

def clip_collate_fn(batch):
    inputs = [item[0] for item in batch]
    label_binary = [item[1] for item in batch]
    label_type = [item[2] for item in batch]
    label_source = [item[3] for item in batch]
    return [inputs, label_binary, label_type, label_source]

def vb_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    label_binary = [item[2] for item in batch]
    label_type = [item[3] for item in batch]
    label_source = [item[4] for item in batch]
    return [text, img, label_binary, label_type, label_source]

###################

def pretrain_clip_collate_fn(batch):
    inputs_first = [item[0] for item in batch]
    inputs_second = [item[1] for item in batch]
    contrastive_label = [item[2] for item in batch]
    return [inputs_first, inputs_second, contrastive_label]
