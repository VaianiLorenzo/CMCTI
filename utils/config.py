import os
from yacs.config import CfgNode as CN


_C = CN()

_C.PATH = CN()
_C.PATH.DIR_PROJECT = "."
_C.PATH.DIR_DATA = os.path.join("data")
_C.PATH.FILE_TRAIN_DATASET = os.path.join(_C.PATH.DIR_DATA, "training", "training_multitask.csv")
_C.PATH.FILE_TEST_DATASET = os.path.join(_C.PATH.DIR_DATA, "test", "test-multitask.csv")

_C.COMET = CN()
_C.COMET.ENABLED = True
_C.COMET.API_KEY = "LiMIt9D5WsCZo294IIYymGhdv"
_C.COMET.PROJECT_NAME = "mami"
_C.COMET.WORKSPACE = "vaianilorenzo"

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 32
_C.DATALOADER.N_WORKERS = 4
_C.DATALOADER.PERCENTAGE_TRAIN = 0.75

_C.MODEL = CN()
_C.MODEL.TYPE = "multitask"  # base - visual_bert - multitask
_C.MODEL.BASELINE_MODALITY = "multimodal"  # image - text
_C.MODEL.CLASS_MODALITY = "cls"  # "VisualBERT Classification Modality (cls or avg)"
_C.MODEL.MASKR_MODALITY = "coco"  # "VisualBERT MASK R-CNN Modality (coco, lvis or both)"
_C.MODEL.MULTITASK_MODALITY = [1, 1, 1]  # Enable/Disable tasks inside the multitask model

_C.TRAINING = CN()
_C.TRAINING.EPOCHS = 25
_C.TRAINING.GAMMA = 1  # Gamma value for optimizer
_C.TRAINING.LR = 1e-5

_C.TEST = CN()
_C.TEST.DIR_CHECKPOINTS = os.path.join("data", f"models")
_C.TEST.FILE_CHECKPOINT = "MAMI_multitask_coco_24.model"


def get_cfg_defaults():
    # Return a clone so that the defaults will not be altered
    return _C.clone()
