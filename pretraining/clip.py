import os

import numpy as np
from sklearn.metrics import f1_score
import torch
from tqdm import tqdm

from torch.nn import CosineEmbeddingLoss
from utils.metrics import binary_acc
import torch.nn.functional as F


def _do_epoch(device, model, dataloader, loss_fn, train=False, optimizer=None, scheduler=None):

    current_loss = 0
    
    if train:
        model.train()
    else:
        model.eval()

    # Iterate over the DataLoader for training data
    current_loss = 0.0

    for data in tqdm(dataloader):

        first, second, labels = data

        first_input_ids = [x["input_ids"][0] for x in first]
        first_attention_mask = [x["attention_mask"][0] for x in first]
        first_pixel_values = [x["pixel_values"][0] for x in first]
        second_input_ids = [x["input_ids"][0] for x in second]
        second_attention_mask = [x["attention_mask"][0] for x in second]
        second_pixel_values = [x["pixel_values"][0] for x in second]

        if train:
            optimizer.zero_grad()
            first_text_features = model.get_text_features(input_ids=torch.stack(first_input_ids).to(device), attention_mask=torch.stack(first_attention_mask).to(device))
            first_image_features = model.get_image_features(pixel_values=torch.stack(first_pixel_values).to(device))
            first_embedding = torch.cat([first_text_features, first_image_features], 1)
            second_text_features = model.get_text_features(input_ids=torch.stack(second_input_ids).to(device), attention_mask=torch.stack(second_attention_mask).to(device))
            second_image_features = model.get_image_features(pixel_values=torch.stack(second_pixel_values).to(device))
            second_embedding = torch.cat([second_text_features, second_image_features], 1)
        else:
            with torch.no_grad():
                first_text_features = model.get_text_features(input_ids=torch.stack(first_input_ids).to(device), attention_mask=torch.stack(first_attention_mask).to(device))
                first_image_features = model.get_image_features(pixel_values=torch.stack(first_pixel_values).to(device))
                first_embedding = torch.cat([first_text_features, first_image_features], 1)
                second_text_features = model.get_text_features(input_ids=torch.stack(second_input_ids).to(device), attention_mask=torch.stack(second_attention_mask).to(device))
                second_image_features = model.get_image_features(pixel_values=torch.stack(second_pixel_values).to(device))
                second_embedding = torch.cat([second_text_features, second_image_features], 1)

        labels = torch.tensor(labels).to(device).float()
        print("LABELS:", labels)

        loss = loss_fn(first_embedding, second_embedding, labels)

        if train:
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_loss += loss.item()

    return current_loss / len(dataloader)


def pretrain_model(cfg, model, device, n_epochs, optimizer, scheduler, train_dataloader, val_dataloader,
                path_dir_checkpoint, comet_exp):

    loss_fn = CosineEmbeddingLoss()

    best_score = 100
    best_epoch = 0
    for epoch in range(0, n_epochs):
        print(f'Starting epoch {epoch + 1}')

        train_loss = _do_epoch(device, model,
                                train_dataloader,
                                loss_fn,
                                train=True,
                                optimizer=optimizer,
                                scheduler=scheduler)

        print("LR:", scheduler.get_last_lr())
        print("Train Loss:", train_loss)

        ##### Validation #####
        val_loss = _do_epoch(device, model, val_dataloader, loss_fn)

        # saving as checkpoint
        if val_loss < best_score:
            file_name = f"best.model"
            torch.save(model, os.path.join(path_dir_checkpoint, file_name))
            best_score = val_loss
            best_epoch = epoch+1

        print("Validation Loss:", val_loss)
        
        f = open(f"logs/pretraining_log.txt", "a+")
        f.write("Epoch " + str(epoch + 1) + ":\n")
        f.write("\tTrain loss:\t\t%.8f \n" % train_loss)
        f.write("\tValidation loss:\t%.8f \n" % val_loss)
        f.close()

        if cfg.COMET.ENABLED:
            comet_exp.log_metrics(
                {"Loss": train_loss},
                prefix="Train",
                step=(epoch + 1),
            )
            comet_exp.log_metrics(
                {"Loss": val_loss},
                prefix="Validation",
                step=(epoch + 1),
            )

    print(f"\nBest model found at epoch {best_epoch} with an average validation F1 score of: {best_score}")
    f = open(f"logs/pretraining_log.txt", "a+")
    f.write("\n________________\nBest model found at epoch" + str(best_epoch) +  "with a validation loss of: " + str(best_score) + "\n________________\n")
    f.close()
