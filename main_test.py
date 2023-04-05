import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import read_config

from sklearn.decomposition import PCA
import plotly.express as px

os.environ["TOKENIZERS_PARALLELISM"] = "false"

torch.multiprocessing.set_sharing_strategy('file_system')

path_output_file = os.path.join("data", "output.csv")
path_submission_file = os.path.join("data", "submission.csv")

if __name__ == "__main__":
    cfg = read_config()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(cfg.TEST.DIR_CHECKPOINTS, cfg.TEST.FILE_CHECKPOINT)
    model = torch.load(os.path.join(cfg.TEST.DIR_CHECKPOINTS, cfg.TEST.FILE_CHECKPOINT))
    if cfg.TEST.RETURN_EMBEDDINGS:
        model.return_embeddings = True
    else:
        model.return_embeddings = False
    model = model.to(device)
    model.eval()


    test_dataloader = torch.load(os.path.join("data", "dataloaders", f"test_{cfg.MODEL.TYPE}_dataloader.bkp"))
    test_dataloader.shuffle = False
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

    # post-processed predictions
    cleaned_list_type_outputs = []
    cleaned_list_source_modality_outputs = []

    #visualization
    embedding_list = []

    with torch.no_grad():
        for index, data in enumerate(tqdm(test_dataloader)):
            if cfg.MODEL.TYPE == "visual_bert":
                texts, images, binary_targets, type_targets, source_modality_targets = data
                predicted = model(texts, images)
                length = len(images)
            elif cfg.MODEL.TYPE == "clip-base" or cfg.MODEL.TYPE == "clip-large":
                inputs, binary_targets, type_targets, source_modality_targets = data
                predicted = model(inputs)
                length = len(inputs)
            elif cfg.MODEL.TYPE == "baseline":
                texts, images, binary_targets, type_targets, source_modality_targets = data
                predicted = model(texts, images)
                length = len(images)

            if cfg.TEST.RETURN_EMBEDDINGS:
                out_binary, out_type, out_source, embeddings = predicted
                embedding_list.append(embeddings)
            else:
                out_binary, out_type, out_source = predicted

            out_binary = torch.sigmoid(out_binary).round().int().tolist()

            type_prob = torch.sigmoid(out_type)
            type_pred = [sample_type_prob.round().int().tolist() for sample_type_prob in type_prob]
            cleaned_type_pred = [[0,0,0,0] if a[0] == 0 else b for a,b in zip(out_binary, type_pred)]

            # When there is no redundant label in task B, the binary label is derived from the others
            if not cfg.MODEL.USE_REDUNDANT_LABELS:
                list_type_misogyny.extend(torch.round(type_prob).any(dim=1).int().tolist())

            source_prob = F.softmax(out_source, dim=1)
            source_pred = [sp.tolist().index(max(sp.tolist())) for sp in source_prob]
            cleaned_source_pred = [0 if a[0] == 0 else b for a,b in zip(out_binary, source_pred)]

            for i in range(length):
                #names.append(os.path.basename(images[i]))
                list_binary_outputs.append(out_binary[i][0])

                if cfg.MODEL.USE_REDUNDANT_LABELS:
                    list_type_misogyny.append(type_pred[i][0])
                    list_type_shaming.append(type_pred[i][1])
                    list_type_stereotype.append(type_pred[i][2])
                    list_type_objectification.append(type_pred[i][3])
                    list_type_violence.append(type_pred[i][4])
                else:
                    list_type_shaming.append(type_pred[i][0])
                    list_type_stereotype.append(type_pred[i][1])
                    list_type_objectification.append(type_pred[i][2])
                    list_type_violence.append(type_pred[i][3])
                    
                list_source_modality_outputs.append(source_pred[i])
                cleaned_list_source_modality_outputs.append(cleaned_source_pred[i])

                # Register targets for metrics calculation
                binary_ground_truth.append(binary_targets[i])
                source_modality_ground_truth.append(torch.argmax(torch.tensor(source_modality_targets[i])).item())

            # Register type targets and predictions in lists for metrics calculation
            list_type_outputs.extend(type_pred)
            cleaned_list_type_outputs.extend(cleaned_type_pred)
            type_ground_truth.extend(type_targets)

    # --------- Calculate metrics ---------
    # F1 Task A
    binary_precision = precision_score(np.array(binary_ground_truth),
                            torch.round(torch.sigmoid(torch.tensor(list_binary_outputs))).numpy())
    binary_recall = recall_score(np.array(binary_ground_truth),
                            torch.round(torch.sigmoid(torch.tensor(list_binary_outputs))).numpy())
    binary_f1 = f1_score(np.array(binary_ground_truth),
                            torch.round(torch.sigmoid(torch.tensor(list_binary_outputs))).numpy())
    
    
    # F1 Task B
    type_precision = precision_score(np.array(type_ground_truth), np.array(list_type_outputs), average="macro")
    type_recall = recall_score(np.array(type_ground_truth), np.array(list_type_outputs), average="macro")
    type_f1_macro = f1_score(np.array(type_ground_truth), np.array(list_type_outputs), average="macro")
    type_f1_none = f1_score(np.array(type_ground_truth), np.array(list_type_outputs), average=None)
    type_f1 = [type_f1_macro, type_f1_none]

    cleaned_type_f1_macro = f1_score(np.array(type_ground_truth), np.array(cleaned_list_type_outputs), average="macro")
    cleaned_type_f1_none = f1_score(np.array(type_ground_truth), np.array(cleaned_list_type_outputs), average=None)
    cleaned_type_f1 = [cleaned_type_f1_macro, cleaned_type_f1_none]
    

    # F1 Task C
    source_modality_precision = precision_score(np.array(source_modality_ground_truth),
                                        torch.tensor(list_source_modality_outputs).numpy(), average="macro")
    source_modality_recall = recall_score(np.array(source_modality_ground_truth),
                                        torch.tensor(list_source_modality_outputs).numpy(), average="macro")
    source_modality_f1_macro = f1_score(np.array(source_modality_ground_truth),
                                        torch.tensor(list_source_modality_outputs).numpy(), average="macro")
    source_modality_f1_none = f1_score(np.array(source_modality_ground_truth),
                                        torch.tensor(list_source_modality_outputs).numpy(), average=None)
    source_modality_f1 = [source_modality_f1_macro, source_modality_f1_none]

    cleaned_source_modality_f1_macro = f1_score(np.array(source_modality_ground_truth),
                                        torch.tensor(cleaned_list_source_modality_outputs).numpy(), average="macro")
    cleaned_source_modality_f1_none = f1_score(np.array(source_modality_ground_truth),
                                        torch.tensor(cleaned_list_source_modality_outputs).numpy(), average=None)
    cleaned_source_modality_f1 = [cleaned_source_modality_f1_macro, cleaned_source_modality_f1_none]

    print("Test Precision Binary: ", binary_precision)
    print("Test Recall Binary: ", binary_recall)
    print("Test F1 Binary: ", binary_f1)
    print("Test Precision Type: ", type_precision)
    print("Test Recall Type: ", type_recall)
    print("Test F1 Type: ", type_f1)
    print("CLEANED Test F1 Type: ", cleaned_type_f1)
    print("Test Precision Source Modality: ", source_modality_precision)
    print("Test Recall Source Modality: ", source_modality_recall)
    print("Test F1 Source Modality: ", source_modality_f1)
    print("CLEANED Test F1 Source Modality: ", cleaned_source_modality_f1)

    # --------- Save results ---------
    df = pd.DataFrame({
        #"names": names,
        "binary": list_binary_outputs,
        "type_misogyny": list_type_misogyny,
        "shaming": list_type_shaming,
        "stereotype": list_type_stereotype,
        "objectification": list_type_objectification,
        "violence": list_type_violence,
        "source": list_source_modality_outputs
    })
    
    df.to_csv(path_output_file, sep="\t", header=True, index=False)
    
    submission_df = df.drop(columns=["binary", "source"])
    submission_df.to_csv(path_submission_file, sep="\t", header=False, index=False)


    if cfg.TEST.RETURN_EMBEDDINGS:
        label_names = ["None", "Text-only", "Image-only", "Either", "Both"]
        colors = [label_names[i] for i in source_modality_ground_truth]
        embedding_list = torch.stack([e for element in embedding_list for e in element]).detach().cpu()

        pca = PCA(n_components=4)
        pca.fit(embedding_list)
        print(pca.explained_variance_ratio_)
        components = pca.transform(embedding_list)

        fig = px.scatter(
            components,
            x=0,
            y=1,
            color=colors
        )
        fig.write_image(f"{cfg.MODEL.TYPE}_test_pca_visualization.png")
        


    
