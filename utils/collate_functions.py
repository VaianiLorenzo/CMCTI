def base_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return [text, img, label]

def base_test_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    path = [item[2] for item in batch]
    return [text, img, path]

def vb_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return [text, img, label]

def vb_test_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    return [text, img]

def multitask_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    binary_label = [item[2] for item in batch]
    source_modality_label = [item[3] for item in batch]
    return [text, img, binary_label, source_modality_label]

def multitask_test_collate_fn(batch):
    text = [item[0] for item in batch]
    img = [item[1] for item in batch]
    return [text, img]
