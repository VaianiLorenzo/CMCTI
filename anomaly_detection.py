import pandas as pd
import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm
import numpy as np

# import OCSVM, ECOD, ABOD, COPOD, MAD, SOS, QMCD, KDE, PCA, MCD, KNN, LUNAR
from pyod.models.ocsvm import OCSVM
from pyod.models.ecod import ECOD
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.sos import SOS
from pyod.models.qmcd import QMCD
from pyod.models.kde import KDE
from pyod.models.pca import PCA
from pyod.models.mcd import MCD
from pyod.models.knn import KNN
from pyod.models.lunar import LUNAR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


path_train_dataset = os.path.join("data", "training", "training_multitask_cleaned.csv")
path_test_dataset = os.path.join("data", "test", "test-multitask.csv")

df_train = pd.read_csv(path_train_dataset, sep=";")
df_test = pd.read_csv(path_test_dataset, sep=";")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name  = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
#model = CLIPModel.from_pretrained(model_name)
model = torch.load(os.path.join("data", "checkpoints", "pretrained_clip", "best.model"))
model.to(device)
model.eval()

sim_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}

#train_text_embeds = []
#train_image_embeds = []

train_embeddings = []
train_labels = []

# iter df rows
for index, row in tqdm(df_train.iterrows(), total=len(df_train)):
    text = row["Text Transcription"]
    image_path = row["file_name"]
    label = 1*row["text"] + 2*row["image"] + 3*row["either"] + 4*row["both"]
    train_labels.append(label)

    # load image
    image = Image.open(os.path.join("data", "training", image_path))

    inputs = processor(text=text, images=image, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    inputs = inputs.to(device)
    outputs = model(**inputs)

    text_embedding = outputs.text_embeds[0]
    image_embedding = outputs.image_embeds[0]

    #train_text_embeds.append(text_embedding.cpu().detach().numpy())
    #train_image_embeds.append(image_embedding.cpu().detach().numpy())
    train_embeddings.append(torch.cat((text_embedding, image_embedding), dim=0).cpu().detach().numpy())

    # cosine similarity between text and image
    similarity = torch.cosine_similarity(text_embedding, image_embedding, dim=0)

    sim_dict[label].append(similarity.item())

    if index == 100:
        break


print("L:", train_labels[0], train_labels[14])
print(train_embeddings[0])
print(train_embeddings[14])


for k, v in sim_dict.items():
    print(k, np.mean(v), np.std(v), np.min(v), np.max(v))


sim_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}

test_embeddings = []
test_labels = []

for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
    text = row["Text Transcription"]
    image_path = row["file_name"]
    label = 1*row["text"] + 2*row["image"] + 3*row["either"] + 4*row["both"]
    test_labels.append(label)

    # load image
    image = Image.open(os.path.join("data", "test", image_path))

    inputs = processor(text=text, images=image, return_tensors="pt", padding="max_length", max_length=77, truncation=True)
    inputs = inputs.to(device)
    outputs = model(**inputs)

    text_embedding = outputs.text_embeds[0]
    image_embedding = outputs.image_embeds[0]

    test_embeddings.append(torch.cat((text_embedding, image_embedding), dim=0).cpu().detach().numpy())

    # cosine similarity between text and image
    similarity = torch.cosine_similarity(text_embedding, image_embedding, dim=0)

    sim_dict[label].append(similarity.item())

    if index == 100:
        break


for k, v in sim_dict.items():
    print(k, np.mean(v), np.std(v), np.min(v), np.max(v))

train_embeddings = np.array(train_embeddings)
test_embeddings = np.array(test_embeddings)

clfs = [OCSVM(), ECOD(), ABOD(), COPOD(), SOS(), QMCD(), KDE(), PCA(), MCD(), KNN(), LUNAR()]
names = ["OCSVM", "ECOD", "ABOD", "COPOD", "SOS", "QMCD", "KDE", "PCA", "MCD", "kNN", "LUNAR"]
test_labels = [1 if l == 4 else 0 for l in test_labels]

for name, clf in zip(names, clfs):
    print("METHOD:", name)

    clf.fit(train_embeddings)

    # get outlier scores
    y_train_scores = clf.decision_scores_  # raw outlier scores on the train data
    y_train_predicted = clf.labels_  # binary labels (0: inliers, 1: outliers) of the train data
    y_test_scores = clf.decision_function(test_embeddings)  # predict raw outlier scores on test
    y_test_predicted = clf.predict(test_embeddings)  # predict binary labels (0 or 1) on test

    print("\tAccuracy: ", accuracy_score(test_labels, y_test_predicted))
    print("\tPrecision: ", precision_score(test_labels, y_test_predicted))
    print("\tRecall: ", recall_score(test_labels, y_test_predicted))
    print("\tF1: ", f1_score(test_labels, y_test_predicted))
