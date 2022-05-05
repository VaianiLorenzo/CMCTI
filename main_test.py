import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import os

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
    # Task B labels
    list_type_misogyny = []
    list_type_shaming = []
    list_type_stereotype = []
    list_type_objectification = []
    list_type_violence = []
    # Task C labels
    list_source_modality_outputs = []
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
