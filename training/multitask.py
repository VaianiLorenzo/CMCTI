import os

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from tqdm import tqdm

from utils.metrics import binary_acc


def _do_epoch(device, model, dataloader, train=False, optimizer=None, scheduler=None):
    binary_loss_function = BCEWithLogitsLoss()
    source_modality_loss_function = CrossEntropyLoss()

    # Iterate over the DataLoader for training data
    current_loss = 0.0
    n_samples = 0
    list_binary_outputs = []
    binary_ground_truth = []
    list_source_modality_outputs = []
    source_modality_ground_truth = []
    model.train()
    for data in tqdm(dataloader):
        # Get and prepare inputs
        texts, images, binary_targets, source_modality_targets = data
        binary_targets = torch.tensor(binary_targets).to(device).float()
        source_modality_targets = torch.tensor(source_modality_targets).to(device).float()

        # Perform forward pass
        if train:
            optimizer.zero_grad()
            out_binary, out_source = model(texts, images)
        else:
            with torch.no_grad():
                out_binary, out_source = model(texts, images)

        out_binary = out_binary.squeeze()

        # update lists for accuracy computation
        list_binary_outputs.extend(list(out_binary))
        list_source_modality_outputs.extend(list(out_source))

        binary_tar = [t.item() for t in binary_targets]
        binary_ground_truth.extend(binary_tar)

        source_modality_tar = [torch.argmax(t).item() for t in source_modality_targets]
        source_modality_ground_truth.extend(source_modality_tar)

        # compute loss
        binary_loss = binary_loss_function(out_binary, binary_targets)
        source_modality_loss = source_modality_loss_function(out_source, source_modality_targets)
        loss = binary_loss + source_modality_loss

        if train:
            # Perform backward pass
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            # Perform optimization
            optimizer.step()
            scheduler.step()

        current_loss += loss.item() * dataloader.batch_size
        n_samples += len(binary_targets)

    epoch_loss = current_loss / n_samples
    acc = binary_acc(torch.tensor(list_binary_outputs), torch.tensor(binary_ground_truth))
    f1 = f1_score(np.array(binary_ground_truth),
                  torch.round(torch.sigmoid(torch.tensor(list_binary_outputs))).numpy())

    return epoch_loss, acc, f1


def train_model(cfg, model, device, n_epochs, optimizer, scheduler, train_dataloader, val_dataloader,
                          path_dir_checkpoint, comet_exp):
    for epoch in range(0, n_epochs):
        print(f'Starting epoch {epoch + 1}')

        train_loss, train_acc, train_f1 = _do_epoch(device, model, train_dataloader, train=True, optimizer=optimizer,
                                                    scheduler=scheduler)

        print("LR:", scheduler.get_last_lr())
        print('Loss after epoch %5d: %.8f' % (epoch + 1, train_loss / len(train_dataloader)))
        print("Train Accuracy: ", train_acc)
        print("Train F1: ", train_f1)

        # saving as checkpoint
        file_name = f"MAMI_multitask_{cfg.MODEL.MASKR_MODALITY}_{epoch}.model"
        torch.save(model, os.path.join(path_dir_checkpoint, file_name))

        ##### Validation #####
        val_loss, val_acc, val_f1 = _do_epoch(device, model, val_dataloader)

        print("Validation Loss:", val_loss)
        print("Validation Accuracy: ", val_acc)
        print("Validation F1: ", val_f1)

        f = open("log_file.txt", "a+")
        f.write("Epoch " + str(epoch + 1) + ":\n")
        f.write("\tTrain loss:\t\t%.8f \n" % train_loss)
        f.write("\tTrain ACCURACY:\t" + str(train_acc) + "\n")
        f.write("\tTrain F1:\t" + str(train_f1) + "\n")
        f.write("\tValidation loss:\t%.8f \n" % val_loss)
        f.write("\tValidation ACCURACY:\t" + str(val_acc) + "\n")
        f.write("\tValidation F1:\t" + str(val_f1) + "\n")
        f.close()

        if cfg.COMET.ENABLED:
            comet_exp.log_metrics(
                {"Loss": train_loss},
                prefix="Train",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"Accuracy": train_acc},
                prefix="Train",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"F1": train_f1},
                prefix="Train",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"Loss": val_loss},
                prefix="Validation",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"Accuracy": val_acc},
                prefix="Validation",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"F1": val_f1},
                prefix="Validation",
                step=(epoch + 1),
            )
