from email.mime import image
import cv2
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPTextConfig
from models.mlp import MLP


class clip_model(nn.Module):

    def __init__(self, device=None, multitask_mod=[1, 1, 1], use_redundant_labels=True, return_embeddings=False, pretrained=False, model_name="clip-base"):
        super().__init__()

        assert multitask_mod != [0, 0, 0], "At least one modality must be active"

        if multitask_mod is None:
            multitask_mod = [1, 1, 1]

        self.multitask_mod = multitask_mod
        
        self.device = device

        if model_name == "clip-base":
            embedding_size = 1024
            if pretrained:
                self.clip = torch.load("data/checkpoints/pretrained_clip-base/best.model")
            else:
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        else:
            embedding_size = 1536
            if pretrained:
                self.clip = torch.load("data/checkpoints/pretrained_clip_large/best.model")
            else:
                self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        self.clip.to(self.device)

        # instantiate MLPs
        self.binary_classifier = MLP(input_dim=embedding_size, output_dim=1)
        self.binary_classifier = self.binary_classifier.to(self.device)

        self.source_classifier = MLP(input_dim=embedding_size, output_dim=5)
        self.source_classifier = self.source_classifier.to(device)

        n_type_labels = 5 if use_redundant_labels else 4
        self.multilabel_classifier = MLP(input_dim=embedding_size, output_dim=n_type_labels)
        self.multilabel_classifier = self.multilabel_classifier.to(device)

        self.return_embeddings = return_embeddings

    def forward(self, inputs):
        ids = []
        mask = []
        pixels = []
        for e in inputs:
            ids.append(e["input_ids"][0])
            mask.append(e["attention_mask"][0])
            pixels.append(e["pixel_values"][0])

        text_features = self.clip.get_text_features(input_ids=torch.stack(ids).to(self.device), attention_mask=torch.stack(mask).to(self.device))
        image_features = self.clip.get_image_features(pixel_values=torch.stack(pixels).to(self.device))

        out_embeddings = torch.cat([text_features, image_features], 1)
    
        # Enable gradient calculation of each head only if the relative modality is enabled
        if self.multitask_mod[0] == 1:
            binary_pred = self.binary_classifier(out_embeddings)
        else:
            with torch.no_grad():
                binary_pred = self.binary_classifier(out_embeddings)

        if self.multitask_mod[1] == 1:
            multilabel_pred = self.multilabel_classifier(out_embeddings)
        else:
            with torch.no_grad():
                multilabel_pred = self.multilabel_classifier(out_embeddings)

        if self.multitask_mod[2] == 1:
            source_pred = self.source_classifier(out_embeddings)
        else:
            with torch.no_grad():
                source_pred = self.source_classifier(out_embeddings)

        if self.return_embeddings:
            return binary_pred, multilabel_pred, source_pred, out_embeddings
        else:
            return binary_pred, multilabel_pred, source_pred
