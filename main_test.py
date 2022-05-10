import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import read_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.multiprocessing.set_sharing_strategy('file_system')

path_output_file = os.path.join("data", "submission_file.csv")

if __name__ == "__main__":
    cfg = read_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = torch.load(os.path.join(cfg.TEST.DIR_CHECKPOINTS, cfg.TEST.FILE_CHECKPOINT))
    model = model.to(device)
    model.eval()

    test_dataloader = torch.load(os.path.join("data", "dataloaders", f"test_{cfg.MODEL.TYPE}_dataloader.bkp"))
    print("Test dataloader length:", len(test_dataloader))

    preds = []
    names = []
    # Task A labels
    list_binary_outputs = []
    binary_ground_truth = []
    # Task B labels
    list_type_misogyny = []
    list_type_shaming = []
    list_type_stereotype = []
    list_type_objectification = []
    list_type_violence = []
    list_type_outputs = []
    type_ground_truth = []
    # Task C labels
    list_source_modality_outputs = []
    source_modality_ground_truth = []
    with torch.no_grad():
        for index, data in enumerate(tqdm(test_dataloader)):
            if cfg.MODEL.TYPE == "multitask":
                texts, images, binary_targets, type_targets, source_modality_targets = data

                predicted = model(texts, images)

                out_binary, out_type, out_source = predicted

                out_binary = torch.sigmoid(out_binary).round().int().tolist()

                type_prob = torch.sigmoid(out_type)
                type_pred = [sample_type_prob.round().int().tolist() for sample_type_prob in type_prob]

                source_prob = F.softmax(out_source, dim=1)
                source_pred = [sp.tolist().index(max(sp.tolist())) for sp in source_prob]
                for i in range(len(images)):
                    names.append(os.path.basename(images[i]))
                    list_binary_outputs.append(out_binary[i][0])

                    list_type_misogyny.append(type_pred[i][0])
                    list_type_shaming.append(type_pred[i][1])
                    list_type_stereotype.append(type_pred[i][2])
                    list_type_objectification.append(type_pred[i][3])
                    list_type_violence.append(type_pred[i][4])

                    list_source_modality_outputs.append(source_pred[i])

                    # Register targets for metrics calculation
                    binary_ground_truth.append(binary_targets[i])
                    source_modality_ground_truth.append(torch.argmax(torch.tensor(source_modality_targets[i])).item())

                # Register type targets and predictions in lists for metrics calculation
                list_type_outputs.extend(type_pred)
                type_ground_truth.extend(type_targets)

            else:
                if cfg.MODEL.TYPE == "base":
                    texts, images, path = data
                elif cfg.MODEL.TYPE == "visual_bert":
                    texts, path = data
                    images = path

                predicted = model(texts, images)
                for p, n in zip(torch.sigmoid(predicted).tolist(), path):
                    preds.append(int(p >= 0.5))
                    names.append(n[5:])

    if cfg.MODEL.TYPE == "multitask":
        # --------- Calculate metrics ---------
        # F1 Task A
        binary_f1 = f1_score(np.array(binary_ground_truth),
                             torch.round(torch.sigmoid(torch.tensor(list_binary_outputs))).numpy())

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

        print("Test F1 Binary: ", binary_f1)
        print("Test F1 Type: ", type_f1)
        print("Test F1 Source Modality: ", source_modality_f1)

        # --------- Save results ---------
        df = pd.DataFrame({
            "names": names,
            "binary": list_binary_outputs,
            "type_misogyny": list_type_misogyny,
            "shaming": list_type_shaming,
            "stereotype": list_type_stereotype,
            "objectification": list_type_objectification,
            "violence": list_type_violence,
            "source": list_source_modality_outputs
        })
    else:
        df = pd.DataFrame({
            "names": names,
            "predictions": preds
        })
    df.to_csv(path_output_file, sep=";", header=True, index=False)
