import torch
import torch.nn as nn


class MultitaskLossA(nn.Module):
    def __init__(self, alpha: float = 1):
        super().__init__()

        self.alpha = alpha
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.cross_entropy_fn = nn.CrossEntropyLoss()

    def forward(self, y_pred_binary: torch.Tensor, y_pred_type: torch.Tensor, y_pred_source: torch.Tensor,
                y_true_binary: torch.Tensor, y_true_type: torch.Tensor, y_true_source: torch.Tensor) -> torch.Tensor:
        binary_loss = self.bce_fn(y_pred_binary, y_true_binary)
        type_loss = self.bce_fn(y_pred_type, y_true_type)
        source_loss = self.cross_entropy_fn(y_pred_source, y_true_source)

        pred_binary_class = torch.round(torch.sigmoid(y_pred_binary))
        pred_type_class = torch.round(torch.sigmoid(y_pred_type))
        pred_source_prob = torch.softmax(y_pred_source, dim=1)
        pred_source_class = torch.argmax(pred_source_prob, dim=1)

        source_consistency_loss = 0
        for binary_c, source_c in zip(pred_binary_class, pred_source_class):
            if (binary_c == 0 and source_c > 0) or (binary_c > 0 and source_c == 0):
                source_consistency_loss += 1
        source_consistency_loss = source_consistency_loss / len(pred_binary_class)

        type_consistency_loss = 0
        for binary_c, type_c in zip(pred_binary_class, pred_type_class[:, 0]):
            if (binary_c == 0 and type_c > 0) or (binary_c > 0 and type_c == 0):
                type_consistency_loss += 1
        type_consistency_loss = type_consistency_loss / len(pred_binary_class)

        return binary_loss + type_loss + source_loss + self.alpha * source_consistency_loss + \
               self.alpha * type_consistency_loss
