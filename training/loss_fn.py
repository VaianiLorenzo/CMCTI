from xmlrpc.client import boolean
import torch
import torch.nn as nn
import numpy as np


class MultitaskLossA(nn.Module):
    def __init__(self, multitask_mod: list, type_weights: torch.Tensor = None, source_weights: torch.Tensor = None,
                 alpha: float = 1, balanced: boolean = False, consistencyAB = True, consistencyAC = True):
        super().__init__()

        assert multitask_mod != [0, 0, 0], "At least one modality must be active"

        self.alpha = alpha
        self.balanced = balanced
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.consistencyAB = consistencyAB
        self.consistencyAC = consistencyAC
        if balanced:
            self.bce_type_fn = [nn.BCEWithLogitsLoss(pos_weight=type_weights[0]),
                                nn.BCEWithLogitsLoss(pos_weight=type_weights[1]),
                                nn.BCEWithLogitsLoss(pos_weight=type_weights[2]),
                                nn.BCEWithLogitsLoss(pos_weight=type_weights[3]),
                                nn.BCEWithLogitsLoss(pos_weight=type_weights[4])]
            self.cross_entropy_fn = nn.CrossEntropyLoss(weight=source_weights)
        else:
            self.cross_entropy_fn = nn.CrossEntropyLoss()
        self.multitask_mod = multitask_mod

    def forward(self, y_pred_binary: torch.Tensor, y_pred_type: torch.Tensor, y_pred_source: torch.Tensor,
                y_true_binary: torch.Tensor, y_true_type: torch.Tensor, y_true_source: torch.Tensor) -> torch.Tensor:

        loss = 0
        # Task A
        if self.multitask_mod[0] == 1:
            loss += self.bce_fn(y_pred_binary, y_true_binary)
        # Task B
        if self.multitask_mod[1] == 1:
            if self.balanced:
                tmp_loss = 0
                for i in range(y_pred_type.shape[1]):
                    tmp_loss += self.bce_type_fn[i](y_pred_type[:, i], y_true_type[:, i])
                loss += tmp_loss / len(self.bce_type_fn)
            else:
                loss += self.bce_fn(y_pred_type, y_true_type)
        # Task C
        if self.multitask_mod[2] == 1:
            loss += self.cross_entropy_fn(y_pred_source, y_true_source)

        # Consistency loss between task A and B
        if self.multitask_mod[0] == 1 and self.multitask_mod[1] == 1 and self.consistencyAB:
            pred_binary_class = torch.round(torch.sigmoid(y_pred_binary))
            pred_type_class = torch.round(torch.sigmoid(y_pred_type))

            consistency_loss = 0
            for binary_c, type_c in zip(pred_binary_class, pred_type_class[:, 0]):
                if (binary_c == 0 and type_c > 0) or (binary_c > 0 and type_c == 0):
                    consistency_loss += 1
            consistency_loss = consistency_loss / len(pred_binary_class)

            loss += self.alpha * consistency_loss

        # Consistency loss between task A and C
        if self.multitask_mod[0] == 1 and self.multitask_mod[2] == 1 and self.consistencyAC:
            pred_binary_class = torch.round(torch.sigmoid(y_pred_binary))
            pred_source_prob = torch.softmax(y_pred_source, dim=1)
            pred_source_class = torch.argmax(pred_source_prob, dim=1)

            consistency_loss = 0
            for binary_c, source_c in zip(pred_binary_class, pred_source_class):
                if (binary_c == 0 and source_c > 0) or (binary_c > 0 and source_c == 0):
                    consistency_loss += 1
            consistency_loss = consistency_loss / len(pred_binary_class)

            loss += self.alpha * consistency_loss

        # Consistency loss between task B and C
        if self.multitask_mod[0] == 0 and self.multitask_mod[1] == 1 and self.multitask_mod[2] == 1:
            pred_type_class = torch.round(torch.sigmoid(y_pred_type))
            pred_source_prob = torch.softmax(y_pred_source, dim=1)
            pred_source_class = torch.argmax(pred_source_prob, dim=1)

            consistency_loss = 0
            for type_c, source_c in zip(pred_type_class[:, 0], pred_source_class):
                if (type_c == 0 and source_c > 0) or (type_c > 0 and source_c == 0):
                    consistency_loss += 1
            consistency_loss = consistency_loss / len(pred_type_class)

            loss += self.alpha * consistency_loss

        return loss


class MultitaskLossNoRedundancy(nn.Module):
    def __init__(self, multitask_mod: list, alpha: float = 1, consistencyAB = True, consistencyAC = True):
        super().__init__()

        assert multitask_mod != [0, 0, 0], "At least one modality must be active"

        self.alpha = alpha
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.consistencyAB = consistencyAB
        self.consistencyAC = consistencyAC
        self.cross_entropy_fn = nn.CrossEntropyLoss()
        self.multitask_mod = multitask_mod

    def forward(self, y_pred_binary: torch.Tensor, y_pred_type: torch.Tensor, y_pred_source: torch.Tensor,
                y_true_binary: torch.Tensor, y_true_type: torch.Tensor, y_true_source: torch.Tensor) -> torch.Tensor:
        loss = 0
        # Task A
        if self.multitask_mod[0] == 1:
            loss += self.bce_fn(y_pred_binary, y_true_binary)
        # Task B
        if self.multitask_mod[1] == 1:
            loss += self.bce_fn(y_pred_type, y_true_type)
        # Task C
        if self.multitask_mod[2] == 1:
            loss += self.cross_entropy_fn(y_pred_source, y_true_source)

        # Consistency loss between task A and B
        if self.multitask_mod[0] == 1 and self.multitask_mod[1] == 1 and self.consistencyAB:
            # no_grad() is required because any() is not differentiable and throws errors
            with torch.no_grad():
                pred_binary_class = torch.round(torch.sigmoid(y_pred_binary))
                # Infer the binary class from the predictions of class B
                pred_type_class = torch.round(torch.sigmoid(y_pred_type)).any(dim=1).int()

            consistency_loss = 0
            for binary_c, type_c in zip(pred_binary_class, pred_type_class):
                if (binary_c == 0 and type_c > 0) or (binary_c > 0 and type_c == 0):
                    consistency_loss += 1

            consistency_loss = consistency_loss / len(pred_binary_class)

            loss += self.alpha * consistency_loss

        # Consistency loss between task A and C
        if self.multitask_mod[0] == 1 and self.multitask_mod[2] == 1 and self.consistencyAC:
            pred_binary_class = torch.round(torch.sigmoid(y_pred_binary))
            pred_source_prob = torch.softmax(y_pred_source, dim=1)
            pred_source_class = torch.argmax(pred_source_prob, dim=1)

            consistency_loss = 0
            for binary_c, source_c in zip(pred_binary_class, pred_source_class):
                if (binary_c == 0 and source_c > 0) or (binary_c > 0 and source_c == 0):
                    consistency_loss += 1
            consistency_loss = consistency_loss / len(pred_binary_class)

            loss += self.alpha * consistency_loss

        # Consistency loss between task B and C
        if self.multitask_mod[0] == 0 and self.multitask_mod[1] == 1 and self.multitask_mod[2] == 1:
            # no_grad() is required because any() is not differentiable and throws errors
            with torch.no_grad():
                # Infer the binary class from the predictions of class B
                pred_type_class = torch.round(torch.sigmoid(y_pred_type)).any(dim=1).int()
            pred_source_prob = torch.softmax(y_pred_source, dim=1)
            pred_source_class = torch.argmax(pred_source_prob, dim=1)

            consistency_loss = 0
            for type_c, source_c in zip(pred_type_class, pred_source_class):
                if (type_c == 0 and source_c > 0) or (type_c > 0 and source_c == 0):
                    consistency_loss += 1
            consistency_loss = consistency_loss / len(pred_type_class)

            loss += self.alpha * consistency_loss

        return loss
