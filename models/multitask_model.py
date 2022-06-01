import cv2
import torch
import torch.nn as nn
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from transformers import VisualBertModel

from models.mlp import MLP


class MAMI_multitask_model(nn.Module):

    def __init__(self, vb_model_name='uclanlp/visualbert-nlvr2-coco-pre', class_modality="cls", maskr_modality="coco",
                 device=None, multitask_mod=[1, 1, 1], use_redundant_labels=True):
        super().__init__()

        assert multitask_mod != [0, 0, 0], "At least one modality must be active"

        if multitask_mod is None:
            multitask_mod = [1, 1, 1]
        self.cfg_maskr_lvis = "LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"
        self.cfg_maskr_coco = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

        self.device = device
        self.multitask_mod = multitask_mod
        self.class_modality = class_modality
        self.maskr_modality = maskr_modality
        self.visual_bert = VisualBertModel.from_pretrained(vb_model_name)
        self.visual_bert.to(self.device)

        if maskr_modality == "coco" or maskr_modality == "both":
            cfg_path = self.cfg_maskr_coco
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # ROI HEADS SCORE THRESHOLD
            if device == "cpu":
                cfg['MODEL']['DEVICE'] = 'cpu'  # if you are not using cuda
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

            self.maskr_coco = build_model(cfg)
            checkpointer = DetectionCheckpointer(self.maskr_coco)  # load weights
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.maskr_coco.eval()  # eval mode
        else:
            self.maskr_coco = None

        if maskr_modality == "lvis" or maskr_modality == "both":
            cfg_path = self.cfg_maskr_lvis
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(cfg_path))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # ROI HEADS SCORE THRESHOLD
            if device == "cpu":
                cfg['MODEL']['DEVICE'] = 'cpu'
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_path)

            self.maskr_lvis = build_model(cfg)
            checkpointer = DetectionCheckpointer(self.maskr_lvis)  # load weights
            checkpointer.load(cfg.MODEL.WEIGHTS)
            self.maskr_lvis.eval()  # eval mode
        else:
            self.maskr_lvis = None

        # instantiate MLPs
        self.binary_classifier = MLP(input_dim=768, output_dim=1)
        self.binary_classifier = self.binary_classifier.to(self.device)

        self.source_classifier = MLP(input_dim=768, output_dim=5)
        self.source_classifier = self.source_classifier.to(device)

        n_type_labels = 5 if use_redundant_labels else 4
        self.multilabel_classifier = MLP(input_dim=768, output_dim=n_type_labels)
        self.multilabel_classifier = self.multilabel_classifier.to(device)

    def forward(self, x_text, x_image):
        visual_embeds = None

        if self.maskr_modality == "coco" or self.maskr_modality == "both":
            self.maskr_coco.eval()
            feats_coco = self.calculate_feats_patches(self.maskr_coco, x_image)

            visual_embeds = feats_coco

        if self.maskr_modality == "lvis" or self.maskr_modality == "both":
            self.maskr_lvis.eval()
            feats_lvis = self.calculate_feats_patches(self.maskr_lvis, x_image)

            if visual_embeds is None:
                visual_embeds = feats_lvis
            else:
                for i in range(len(visual_embeds)):
                    visual_embeds[i] = torch.cat((visual_embeds[i], feats_lvis[i]), 0)

        visual_attention_mask = []
        visual_token_type_ids = []

        for i in range(len(visual_embeds)):
            # Cut out patch features if patches are more than 32
            if len(visual_embeds[i]) > 32:
                visual_embeds[i] = visual_embeds[i][:32]

            mask = [1] * len(visual_embeds[i])
            gap = 32 - len(visual_embeds[i])
            for _ in range(gap):
                visual_embeds[i] = torch.cat(
                    (visual_embeds[i], torch.stack([torch.tensor([0] * 1024).to(self.device)])), 0)
                mask.append(0)

            visual_attention_mask.append(torch.tensor(mask).to(self.device))
            visual_token_type_ids.append(torch.tensor([1] * len(visual_embeds[i])).to(self.device))

        visual_embeds = torch.stack(visual_embeds)
        visual_attention_mask = torch.stack(visual_attention_mask)
        visual_token_type_ids = torch.stack(visual_token_type_ids)

        input_ids = [x['input_ids'][0] for x in x_text]
        attention_mask = [x['attention_mask'][0] for x in x_text]
        token_type_ids = [x['token_type_ids'][0] for x in x_text]
        input_ids = torch.stack(input_ids).to(self.device)
        attention_mask = torch.stack(attention_mask).to(self.device)
        token_type_ids = torch.stack(token_type_ids).to(self.device)

        outputs = self.visual_bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                   visual_embeds=visual_embeds, visual_attention_mask=visual_attention_mask,
                                   visual_token_type_ids=visual_token_type_ids)

        outputs_embeddings = outputs.last_hidden_state

        if self.class_modality == "cls":
            out_embeddings = outputs_embeddings[:, 0]
        else:
            list_embed = []
            for i in range(len(outputs_embeddings)):
                average = self.global_average_pooling(outputs_embeddings[i])
                list_embed.append(average)
            out_embeddings = torch.stack(list_embed)

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

        return binary_pred, multilabel_pred, source_pred

    def calculate_feats_patches(self, model, x_image):
        visual_embeds = []

        inputs = []
        for path in x_image:
            # path = "../MAMI/" + path
            image = cv2.imread(path)
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs.append({"image": image, "height": height, "width": width})

        with torch.no_grad():
            images = model.preprocess_image(inputs)  # don't forget to preprocess
            features = model.backbone(images.tensor)  # set of cnn features
            proposals, _ = model.proposal_generator(images, features, None)  # RPN

            for i in range(len(proposals)):
                # features_ = [torch.stack([features[f][i]]) for f in model.roi_heads.box_in_features]
                features_single = {}
                features_ = []
                for f in model.roi_heads.box_in_features:
                    tensor = torch.stack([features[f][i]])
                    features_.append(tensor)
                    features_single[f] = tensor

                box_features = model.roi_heads.box_pooler(features_, [proposals[i].proposal_boxes])
                box_features = model.roi_heads.box_head(box_features)  # features of all 1k candidates
                predictions = model.roi_heads.box_predictor(box_features)
                pred_instances, pred_inds = model.roi_heads.box_predictor.inference(predictions, [proposals[i]])
                pred_instances = model.roi_heads.forward_with_given_boxes(features_single, pred_instances)
                # output boxes, masks, scores, etc
                pred_instances = model._postprocess(pred_instances, inputs,
                                                    images.image_sizes)  # scale box to orig size
                # features of the proposed boxes
                feats = box_features[pred_inds]
                visual_embeds.append(feats)

        return visual_embeds

    def image_mean_pooling(self, model_output):
        result = torch.sum(model_output, 0) / model_output.size()[0]
        return result

    def text_mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def global_average_pooling(self, model_output):
        result = torch.sum(model_output, 0) / model_output.size()[0]
        return result
