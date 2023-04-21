import logging
import os

from comet_ml import Experiment
import torch

from transformers import CLIPModel
from utils.utils import read_config
from pretraining.clip import pretrain_model as pretrain_clip

torch.multiprocessing.set_sharing_strategy('file_system')

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logging.getLogger().setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

path_output_dir = os.path.join("data", "dataloaders")

if __name__ == "__main__":
    cfg = read_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    experiment = None
    if cfg.COMET.ENABLED:
        experiment = Experiment(
            api_key=cfg.COMET.API_KEY,
            project_name=cfg.COMET.PROJECT_NAME,
            workspace=cfg.COMET.WORKSPACE,
        )

    print("Loading train dataloader..")
    name = "pretrain_train_clip_dataloader.bkp"
    if cfg.PRETRAINING.IS_BINARY:
        name = "pretrain_train_clip_binary_dataloader.bkp"
    train_dataloader = torch.load(os.path.join(path_output_dir, name))
    print("Loading val dataloader..")
    name = "pretrain_val_clip_dataloader.bkp"
    if cfg.PRETRAINING.IS_BINARY:
        name = "pretrain_val_clip_binary_dataloader.bkp"
    val_dataloader = torch.load(os.path.join(path_output_dir, name))


    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)

    # Create checkpoint directory if it does not exist
    folder = "pretrained_clip"
    if cfg.PRETRAINING.IS_BINARY:
        folder = "pretrained_clip_binary"
    path_dir_checkpoint = os.path.join("data", "checkpoints", folder)
    if not os.path.isdir(path_dir_checkpoint):
        os.mkdir(path_dir_checkpoint)

    # Configure optimizer
    percentage_epochs_per_step = 0.4
    step_size = cfg.PRETRAINING.EPOCHS * len(train_dataloader) * percentage_epochs_per_step
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.PRETRAINING.LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=cfg.PRETRAINING.GAMMA)

    # Init training log file
    with open("log_file.txt", "a+") as f:
        f.write(
            "START TRAINING - " + str(cfg.PRETRAINING.EPOCHS) + " epochs - LR: " + str(
                cfg.PRETRAINING.LR) + " - gamma: " + str(
                cfg.PRETRAINING.GAMMA) + " - step_size: " + str(
                percentage_epochs_per_step * cfg.PRETRAINING.EPOCHS) + " epochs\n")


    pretrain_clip(cfg=cfg, model=model, device=device, n_epochs=cfg.PRETRAINING.EPOCHS, optimizer=optimizer,
                        scheduler=scheduler, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                        path_dir_checkpoint=path_dir_checkpoint, comet_exp=experiment)
    