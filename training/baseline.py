import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from training.loss_fn import MultitaskLossA, MultitaskLossNoRedundancy
from tqdm import tqdm
import torch.nn.functional as F

from utils.metrics import binary_acc

def _do_epoch(device, model, dataloader, loss_fn, train=False, optimizer=None, scheduler=None):
    if train:
        model.train()
    else:
        model.eval()

    # Iterate over the DataLoader for training data
    current_loss = 0.0
    n_samples = 0
    # Task A labels
    list_binary_outputs = []
    binary_ground_truth = []
    # Task B labels
    list_type_outputs = []
    type_ground_truth = []
    # Task C labels
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

        type_prob = torch.sigmoid(out_type)
        type_pred = [sample_type_prob.round().tolist() for sample_type_prob in type_prob]
        list_type_outputs.extend(type_pred)

        source_prob = F.softmax(out_source, dim=1)
        source_pred = [sp.tolist().index(max(sp.tolist())) for sp in source_prob]
        list_source_modality_outputs.extend(source_pred)

        binary_tar = [t.item() for t in binary_targets]
        binary_ground_truth.extend(binary_tar)

        type_ground_truth.extend(type_targets.tolist())

        source_modality_tar = [torch.argmax(t).item() for t in source_modality_targets]
        source_modality_ground_truth.extend(source_modality_tar)

        # Compute loss
        loss = loss_fn(out_binary, out_type, out_source, binary_targets, type_targets, source_modality_targets)

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

    # F1 Task A
    binary_f1 = f1_score(np.array(binary_ground_truth),
                         torch.round(torch.sigmoid(torch.tensor(list_binary_outputs))).numpy())
    print(binary_f1)
    

    # F1 Task B
    type_f1_macro = f1_score(np.array(type_ground_truth), np.array(list_type_outputs), average="macro")
    type_f1_none = f1_score(np.array(type_ground_truth), np.array(list_type_outputs), average=None)
    type_f1 = [type_f1_macro, type_f1_none]

    # F1 Task C
    source_modality_f1_macro = f1_score(np.array(source_modality_ground_truth),
                                        torch.tensor(list_source_modality_outputs).numpy(), average="macro")
    source_modality_f1_none = f1_score(np.array(source_modality_ground_truth),
                                       torch.tensor(list_source_modality_outputs).numpy(), average=None)
    source_modality_f1 = [source_modality_f1_macro, source_modality_f1_none]

    return epoch_loss, acc, binary_f1, type_f1, source_modality_f1



def train_model(cfg, model, device, n_epochs, optimizer, scheduler, train_dataloader, val_dataloader,
                path_dir_checkpoint, comet_exp):

    # Create the multitask loss
    if not cfg.MODEL.USE_REDUNDANT_LABELS:
        loss_fn = MultitaskLossNoRedundancy(multitask_mod=cfg.MODEL.MULTITASK_MODALITY,
                                            consistencyAB=cfg.TRAINING.CONSISTENCY_AB,
                                            consistencyAC=cfg.TRAINING.CONSISTENCY_AC)
    elif cfg.TRAINING.BALANCED:
        source_weights = torch.Tensor(np.load(cfg.PATH.FILE_SOURCE_WEIGHTS)).to(device)
        type_weights = torch.Tensor(np.load(cfg.PATH.FILE_TYPE_WEIGHTS)).to(device)
        loss_fn = MultitaskLossA(multitask_mod=cfg.MODEL.MULTITASK_MODALITY, type_weights=type_weights,
                                 source_weights=source_weights, balanced=True,
                                 consistencyAB=cfg.TRAINING.CONSISTENCY_AB, consistencyAC=cfg.TRAINING.CONSISTENCY_AC)
    else:
        loss_fn = MultitaskLossA(multitask_mod=cfg.MODEL.MULTITASK_MODALITY, balanced=False,
                                 consistencyAB=cfg.TRAINING.CONSISTENCY_AB, consistencyAC=cfg.TRAINING.CONSISTENCY_AC)

    best_score = 0
    best_epoch = 0
    for epoch in range(0, n_epochs):
        print(f'Starting epoch {epoch + 1}')

        train_loss, train_acc, train_binary_f1, train_type_f1, train_source_modality_f1 = _do_epoch(device, model,
                                                                                                    train_dataloader,
                                                                                                    loss_fn,
                                                                                                    train=True,
                                                                                                    optimizer=optimizer,
                                                                                                    scheduler=scheduler)

        average_train_f1 = (train_binary_f1 * cfg.MODEL.MULTITASK_MODALITY[0] + 
                               train_type_f1[0] * cfg.MODEL.MULTITASK_MODALITY[1] + 
                               train_source_modality_f1[0] * cfg.MODEL.MULTITASK_MODALITY[2]) / sum(cfg.MODEL.MULTITASK_MODALITY)

        print("LR:", scheduler.get_last_lr())
        print('Loss after epoch %5d: %.8f' % (epoch + 1, train_loss / len(train_dataloader)))
        print("Train Accuracy: ", train_acc)
        print("Train F1 Binary: ", train_binary_f1)
        print("Train F1 Type: ", train_type_f1)
        print("Train F1 Source Modality: ", train_source_modality_f1)
        print("Train F1 Average: ", average_train_f1)

        ##### Validation #####
        val_loss, val_acc, val_binary_f1, val_type_f1, val_source_modality_f1 = _do_epoch(device, model, val_dataloader,
                                                                                          loss_fn)

        average_val_f1 = (val_binary_f1 * cfg.MODEL.MULTITASK_MODALITY[0] + 
                            val_type_f1[0] * cfg.MODEL.MULTITASK_MODALITY[1] + 
                            val_source_modality_f1[0] * cfg.MODEL.MULTITASK_MODALITY[2]) / sum(cfg.MODEL.MULTITASK_MODALITY)

        # saving as checkpoint
        if average_val_f1 > best_score:
            file_name = f"best.model"
            torch.save(model, os.path.join(path_dir_checkpoint, file_name))
            best_score = average_val_f1
            best_epoch = epoch+1

        print("Validation Loss:", val_loss)
        print("Validation Accuracy: ", val_acc)
        print("Validation F1 Binary: ", val_binary_f1)
        print("Validation F1 Type: ", val_type_f1)
        print("Validation F1 Source Modality: ", val_source_modality_f1)
        print("Validation F1 Average: ", average_val_f1)

        f = open(f"logs/log_{cfg.MODEL.TYPE}_{cfg.MODEL.BASELINE_MODALITY}_{cfg.MODEL.MULTITASK_MODALITY[0]}{cfg.MODEL.MULTITASK_MODALITY[1]}{cfg.MODEL.MULTITASK_MODALITY[2]}_{cfg.TRAINING.CONSISTENCY_AB}{cfg.TRAINING.CONSISTENCY_AC}.txt", "a+")
        f.write("Epoch " + str(epoch + 1) + ":\n")
        f.write("\tTrain loss:\t\t%.8f \n" % train_loss)
        f.write("\tTrain ACCURACY:\t" + str(train_acc) + "\n")
        f.write("\tTrain Binary F1:\t" + str(train_binary_f1) + "\n")
        f.write("\tTrain Type F1:\t" + str(train_type_f1) + "\n")
        f.write("\tTrain Source Modality F1:\t" + str(train_source_modality_f1) + "\n")
        f.write("\tTrain F1 Average:\t" + str(average_train_f1) + "\n")
        f.write("\tValidation loss:\t%.8f \n" % val_loss)
        f.write("\tValidation ACCURACY:\t" + str(val_acc) + "\n")
        f.write("\tValidation Binary F1:\t" + str(val_binary_f1) + "\n")
        f.write("\tValidation Type F1:\t" + str(val_type_f1) + "\n")
        f.write("\tValidation Source Modality F1:\t" + str(val_source_modality_f1) + "\n")
        f.write("\tValidation F1 Average:\t" + str(average_val_f1) + "\n")
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
                {"F1-binary": train_binary_f1},
                prefix="Train",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"F1-type": train_type_f1},
                prefix="Train",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"F1-source-modality": train_source_modality_f1},
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
                {"F1-binary": val_binary_f1},
                prefix="Validation",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"F1-type": val_type_f1},
                prefix="Validation",
                step=(epoch + 1),
            )

            comet_exp.log_metrics(
                {"F1-source-modality": val_source_modality_f1},
                prefix="Validation",
                step=(epoch + 1),
            )

    print(f"\nBest model found at epoch {best_epoch} with an average validation F1 score of: {best_score}")
    f = open(f"logs/log_{cfg.MODEL.TYPE}_{cfg.MODEL.BASELINE_MODALITY}_{cfg.MODEL.MULTITASK_MODALITY[0]}{cfg.MODEL.MULTITASK_MODALITY[1]}{cfg.MODEL.MULTITASK_MODALITY[2]}_{cfg.TRAINING.CONSISTENCY_AB}{cfg.TRAINING.CONSISTENCY_AC}.txt", "a+")
    f.write("\n________________\nBest model found at epoch" + str(best_epoch) +  "with an average validation F1 score of: " + str(best_score) + "\n________________\n")
    f.close()
