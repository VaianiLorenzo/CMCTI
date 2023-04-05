import logging
import os

from comet_ml import Experiment
import torch

from models.vb_model import vb_model
from models.clip_model import clip_model
from models.baseline_model import baseline_model
from utils.config import get_cfg_defaults
from training.baseline import train_model as train_baseline
from training.clip import train_model as train_clip
from training.vb import train_model as train_vb
from utils.utils import read_config

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

path_output_dir = os.path.join("data", "dataloaders")

if __name__ == "__main__":
    cfg = read_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    experiment = None
    if cfg.COMET.ENABLED:
        experiment = Experiment(
            api_key=cfg.COMET.API_KEY,
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
        )

    print("Loading train dataloader..")
    if cfg.TRAINING.AUGMENTATION:
        train_dataloader = torch.load(os.path.join(path_output_dir, f"augmentation_train_{cfg.MODEL.TYPE}_dataloader.bkp"))
    else:
        train_dataloader = torch.load(os.path.join(path_output_dir, f"train_{cfg.MODEL.TYPE}_dataloader.bkp"))
    print("Loading val dataloader..")
    val_dataloader = torch.load(os.path.join(path_output_dir, f"val_{cfg.MODEL.TYPE}_dataloader.bkp"))

    if cfg.MODEL.TYPE == "baseline":
        model = baseline_model(device=device, modality=cfg.MODEL.BASELINE_MODALITY,
                                    multitask_mod=cfg.MODEL.MULTITASK_MODALITY,
                                    use_redundant_labels=cfg.MODEL.USE_REDUNDANT_LABELS)
        prefix = f"{cfg.MODEL.TYPE}_{cfg.MODEL.BASELINE_MODALITY}"
    elif cfg.MODEL.TYPE == "clip-base" or cfg.MODEL.TYPE == "clip-large":
        model = clip_model(device=device, multitask_mod=cfg.MODEL.MULTITASK_MODALITY,
                                     use_redundant_labels=cfg.MODEL.USE_REDUNDANT_LABELS,
                                     pretrained=cfg.MODEL.PRETRAINED,
                                     model_name=cfg.MODEL.TYPE)
        prefix = f"{cfg.MODEL.TYPE}"
    elif cfg.MODEL.TYPE == "visual_bert":
        model = vb_model(device=device, class_modality=cfg.MODEL.CLASS_MODALITY,
                                     maskr_modality=cfg.MODEL.MASKR_MODALITY,
                                     multitask_mod=cfg.MODEL.MULTITASK_MODALITY,
                                     use_redundant_labels=cfg.MODEL.USE_REDUNDANT_LABELS)
        prefix = f"{cfg.MODEL.TYPE}"
    else:
        raise Exception("Invalid model type")

    if cfg.TRAINING.AUGMENTATION:
        prefix = f"augmentation_{prefix}"
    if cfg.MODEL.PRETRAINED:
        prefix = f"pretrained_{prefix}"

    # Create checkpoint directory if it does not exist
    path_dir_checkpoint = os.path.join("data", "checkpoints", prefix + f"_{cfg.MODEL.MULTITASK_MODALITY[0]}{cfg.MODEL.MULTITASK_MODALITY[1]}{cfg.MODEL.MULTITASK_MODALITY[2]}_Loss{cfg.TRAINING.CONSISTENCY_AB}{cfg.TRAINING.CONSISTENCY_AC}_Bal{cfg.TRAINING.BALANCED}")
    if not os.path.isdir(path_dir_checkpoint):
        os.mkdir(path_dir_checkpoint)
    model.to(device)

    # Configure optimizer
    percentage_epochs_per_step = 0.4
    step_size = cfg.TRAINING.EPOCHS * len(train_dataloader) * percentage_epochs_per_step
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAINING.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=cfg.TRAINING.GAMMA)

    # Init training log file
    with open("log_file.txt", "a+") as f:
        f.write(
            "START TRAINING - " + str(cfg.TRAINING.EPOCHS) + " epochs - LR: " + str(
                cfg.TRAINING.LR) + " - gamma: " + str(
                cfg.TRAINING.GAMMA) + " - step_size: " + str(
                percentage_epochs_per_step * cfg.TRAINING.EPOCHS) + " epochs\n")

    if cfg.MODEL.TYPE == "baseline":
        train_baseline(cfg=cfg, model=model, device=device, n_epochs=cfg.TRAINING.EPOCHS, optimizer=optimizer,
                        scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        path_dir_checkpoint=path_dir_checkpoint, comet_exp=experiment)
    elif cfg.MODEL.TYPE == "clip-base" or cfg.MODEL.TYPE == "clip-large": 
        train_clip(cfg=cfg, model=model, device=device, n_epochs=cfg.TRAINING.EPOCHS, optimizer=optimizer,
                        scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        path_dir_checkpoint=path_dir_checkpoint, comet_exp=experiment)
    elif cfg.MODEL.TYPE == "visual_bert":
        train_vb(cfg=cfg, model=model, device=device, n_epochs=cfg.TRAINING.EPOCHS, optimizer=optimizer,
                        scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        path_dir_checkpoint=path_dir_checkpoint, comet_exp=experiment)
