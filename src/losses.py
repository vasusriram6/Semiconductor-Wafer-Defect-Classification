import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', ignore_index=-100):
        """
        alpha: Tensor of shape (num_classes,) giving weight per class. If None, no weighting.
        gamma: focusing parameter >=0, default 2.0.
        reduction: 'mean', 'sum', or 'none'
        ignore_index: class index to ignore in loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(weight=None, reduction='none', ignore_index=ignore_index)

        if self.alpha is not None:
            if not torch.is_tensor(self.alpha):
                self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
            self.alpha = self.alpha.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, inputs, targets):
        """
        inputs: (batch_size, num_classes) raw logits (no softmax applied)
        targets: (batch_size,) ground truth class indices (long)
        """
        # Compute cross entropy loss per sample
        ce_loss = self.ce_loss(inputs, targets)  # shape: (batch_size,)

        # Calculate pt = exp(-CE)
        pt = torch.exp(-ce_loss)  # pt is probability of true class

        # Compute focal loss modulation
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        # Apply alpha class weights if provided
        if self.alpha is not None:
            at = self.alpha.gather(0, targets)
            focal_loss = at * focal_loss

        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss