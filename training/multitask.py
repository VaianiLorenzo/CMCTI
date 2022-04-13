import os

import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from tqdm import tqdm

from training.loss_fn import MultitaskLossA
from utils.metrics import binary_acc
import torch.nn.functional as F


def _do_epoch(device, model, dataloader, train=False, optimizer=None, scheduler=None):
    # binary_loss_function = BCEWithLogitsLoss()
    # source_modality_loss_function = CrossEntropyLoss()
    loss_fn = MultitaskLossA()

    if train:
        model.train()
    else:
        model.eval()

    # Iterate over the DataLoader for training data
    current_loss = 0.0
    n_samples = 0
    list_binary_outputs = []
    binary_ground_truth = []
    list_source_modality_outputs = []
    source_modality_ground_truth = []
    for data in tqdm(dataloader):
        # Get and prepare inputs
        texts, images, binary_targets, type_targets, source_modality_targets = data
        binary_targets = torch.tensor(binary_targets).to(device).float()
        type_targets = torch.tensor(type_targets).to(device).float()
        source_modality_targets = torch.tensor(source_modality_targets).to(device).float()

        # Perform forward pass
        if train:
            optimizer.zero_grad()
            out_binary, out_type, out_source = model(texts, images)
        else:
            with torch.no_grad():
                out_binary, out_type, out_source = model(texts, images)

        out_binary = out_binary.squeeze()

        # update lists for accuracy computation
        list_binary_outputs.extend(list(out_binary))
        source_prob = F.softmax(out_source, dim=1)
        source_pred = [sp.tolist().index(max(sp.tolist())) for sp in source_prob]
        list_source_modality_outputs.extend(source_pred)

        binary_tar = [t.item() for t in binary_targets]
        binary_ground_truth.extend(binary_tar)

        source_modality_tar = [torch.argmax(t).item() for t in source_modality_targets]
        source_modality_ground_truth.extend(source_modality_tar)

        # Compute loss
        loss = loss_fn(out_binary, out_type, out_source, binary_targets, type_targets, source_modality_targets)

        # # compute loss
        # binary_loss = binary_loss_function(out_binary, binary_targets)
        # print("\tBL", binary_loss)
        # source_modality_loss = source_modality_loss_function(out_source, source_modality_targets)
        # print("\tML", source_modality_loss)
        #
        # binary_pred = torch.round(torch.sigmoid(out_binary))
        # source_prob = F.softmax(out_source, dim=1)
        # source_pred = [sp.tolist().index(max(sp.tolist())) for sp in source_prob]
        # consistency_loss = 0
        # for b_pred, s_pred in zip(binary_pred, source_pred):
        #     if (b_pred == 0 and s_pred > 0) or (b_pred > 0 and s_pred == 0):
        #         consistency_loss += 1
        # consistency_loss = consistency_loss / len(binary_pred)
        # print("\tCL", consistency_loss)
        #
        # loss = binary_loss + source_modality_loss + consistency_loss
        # print("LOSS:", loss)

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
    binary_f1 = f1_score(np.array(binary_ground_truth),
                  torch.round(torch.sigmoid(torch.tensor(list_binary_outputs))).numpy())
    source_modality_f1_macro = f1_score(np.array(source_modality_ground_truth), torch.tensor(list_source_modality_outputs).numpy(), average = "macro")
    source_modality_f1_none = f1_score(np.array(source_modality_ground_truth), torch.tensor(list_source_modality_outputs).numpy(), average = None)
    source_modality_f1 = [source_modality_f1_macro, source_modality_f1_none]

    return epoch_loss, acc, binary_f1, source_modality_f1


def train_model(cfg, model, device, n_epochs, optimizer, scheduler, train_dataloader, val_dataloader,
                          path_dir_checkpoint, comet_exp):
    for epoch in range(0, n_epochs):
        print(f'Starting epoch {epoch + 1}')

        train_loss, train_acc, train_binary_f1, train_source_modality_f1 = _do_epoch(device, model, train_dataloader, train=True, optimizer=optimizer,
                                                    scheduler=scheduler)

        print("LR:", scheduler.get_last_lr())
        print('Loss after epoch %5d: %.8f' % (epoch + 1, train_loss / len(train_dataloader)))
        print("Train Accuracy: ", train_acc)
        print("Train Binary F1: ", train_binary_f1)
        print("Train Source Modality F1: ", train_source_modality_f1)

        # saving as checkpoint
        file_name = f"MAMI_multitask_{cfg.MODEL.MASKR_MODALITY}_{epoch}.model"
        torch.save(model, os.path.join(path_dir_checkpoint, file_name))

        ##### Validation #####
        val_loss, val_acc, val_binary_f1, val_source_modality_f1 = _do_epoch(device, model, val_dataloader)

        print("Validation Loss:", val_loss)
        print("Validation Accuracy: ", val_acc)
        print("Validation F1: ", val_binary_f1)
        print("Validation Source Modality F1: ", val_source_modality_f1)

        f = open("log_file.txt", "a+")
        f.write("Epoch " + str(epoch + 1) + ":\n")
        f.write("\tTrain loss:\t\t%.8f \n" % train_loss)
        f.write("\tTrain ACCURACY:\t" + str(train_acc) + "\n")
        f.write("\tTrain Binary F1:\t" + str(train_binary_f1) + "\n")
        f.write("\tTrain Source Modality F1:\t" + str(train_source_modality_f1) + "\n")
        f.write("\tValidation loss:\t%.8f \n" % val_loss)
        f.write("\tValidation ACCURACY:\t" + str(val_acc) + "\n")
        f.write("\tValidation Binary F1:\t" + str(val_binary_f1) + "\n")
        f.write("\tValidation Source Modality F1:\t" + str(val_source_modality_f1) + "\n")
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
