import torch
import torch.nn as nn
import torch.nn.functional as F
from ....datasets.utils import DataProcessing


def one_hot(index, classes):
    out_idx = torch.arange(classes, device=index.device)
    out_idx = torch.unsqueeze(out_idx, 0)
    index = torch.unsqueeze(index, -1)
    return (index == out_idx).float()


class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss."""

    def __init__(self, loss_weight=1.0, log_weight=0.0, class_weights=None):
        """CrossEntropyLoss.

        Args:
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        self.loss_weight = float(loss_weight)
        self.log_weight = float(log_weight)
        if class_weights is not None:
            self.weight = DataProcessing.get_class_weights(class_weights)
        else:
            self.weight = None


    def forward(self, cls_score, label, ignore_index=-100, avg_factor=None, **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss = F.cross_entropy(cls_score, label, weight=self.weight, ignore_index=ignore_index, reduction='none')

        if self.loss_weight >= 0.:
          loss = loss * self.loss_weight

        if avg_factor:
            return loss.sum() / avg_factor
        else:
            return loss.mean()
