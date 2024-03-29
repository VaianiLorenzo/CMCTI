import os
from yacs.config import CfgNode as CN


_C = CN()

_C.PATH = CN()
_C.PATH.DIR_PROJECT = "."
_C.PATH.DIR_DATA = os.path.join("data")
_C.PATH.FILE_TRAIN_DATASET = os.path.join(_C.PATH.DIR_DATA, "training", "training_multitask_cleaned.csv")
_C.PATH.FILE_TEST_DATASET = os.path.join(_C.PATH.DIR_DATA, "test", "test-multitask.csv")
_C.PATH.FILE_SOURCE_WEIGHTS = os.path.join(_C.PATH.DIR_DATA, "source_weights.npy")
_C.PATH.FILE_TYPE_WEIGHTS = os.path.join(_C.PATH.DIR_DATA, "type_weights.npy")

_C.COMET = CN()
_C.COMET.ENABLED = False
_C.COMET.API_KEY = "LiMIt9D5WsCZo294IIYymGhdv"
_C.COMET.PROJECT_NAME = "mami"
_C.COMET.WORKSPACE = "vaianilorenzo"

_C.DATALOADER = CN()
_C.DATALOADER.PRETRAINING_BATCH_SIZE = 2
_C.DATALOADER.BATCH_SIZE = 16
_C.DATALOADER.N_WORKERS = 8
_C.DATALOADER.PERCENTAGE_TRAIN = 0.75

_C.MODEL = CN()
_C.MODEL.TYPE = "clip-base"  # baseline | clip-base | clip-large | visual_bert
_C.MODEL.BASELINE_MODALITY = "image"  # image - text - multimodal
_C.MODEL.CLASS_MODALITY = "cls"  # "VisualBERT Classification Modality (cls or avg)"
_C.MODEL.MASKR_MODALITY = "coco"  # "VisualBERT MASK R-CNN Modality (coco, lvis or both)"
_C.MODEL.MULTITASK_MODALITY = [1, 0, 1]  # Enable/Disable tasks inside the multitask model
_C.MODEL.USE_REDUNDANT_LABELS = False  # Use an additional label in task B indicating the presence of misogyny
_C.MODEL.PRETRAINED = False  # Use a pretrained model

_C.PRETRAINING = CN()
_C.PRETRAINING.EPOCHS = 300
_C.PRETRAINING.GAMMA = 1.0  # Gamma value for optimizer
_C.PRETRAINING.LR = 1e-5
_C.PRETRAINING.IS_BINARY = True

_C.TRAINING = CN()
_C.TRAINING.EPOCHS = 25
_C.TRAINING.GAMMA = 1.0  # Gamma value for optimizer
_C.TRAINING.LR = 1e-6
_C.TRAINING.BALANCED = False
_C.TRAINING.CONSISTENCY_AB = True
_C.TRAINING.CONSISTENCY_AC = True
_C.TRAINING.AUGMENTATION = True

_C.TEST = CN()
#_C.TEST.DIR_CHECKPOINTS = os.path.join("data", "checkpoints", "visual_bert_001_LossTrueTrue_BalFalse")
_C.TEST.DIR_CHECKPOINTS = os.path.join("data", "checkpoints", "augmentation_clip-base_101_LossTrueTrue_BalFalse")
_C.TEST.FILE_CHECKPOINT = "best.model"
_C.TEST.RETURN_EMBEDDINGS = True


def get_cfg_defaults():
    # Return a clone so that the defaults will not be altered
    return _C.clone()
