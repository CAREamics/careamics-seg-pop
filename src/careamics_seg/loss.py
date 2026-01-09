"""Segmentation losses."""

from typing import Callable

import torch
from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F


class DiceLoss(Module):
    """Dice loss for binary and multi-class segmentation.
    
    For binary segmentation (inputs with 1 channel), applies sigmoid activation.
    For multi-class segmentation (inputs with >1 channels), applies softmax activation
    and computes Dice coefficient per class, then averages across classes.
    
    Parameters
    ----------
    weight : Tensor, optional
        A manual rescaling weight given to each class.
    include_background : bool, default=True
        Whether to include the background class (class 0) in the loss calculation.
    """
    
    def __init__(self, weight=None, include_background=True):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.include_background = include_background

    def forward(self, inputs, targets, smooth=1):
        """Compute Dice loss.
        
        Parameters
        ----------
        inputs : Tensor
            Predicted logits of shape (B, C, H, W) or (B, C, D, H, W)
            where C is the number of classes (C=1 for binary).
        targets : Tensor
            Ground truth of shape (B, H, W) or (B, D, H, W) with class indices,
            or (B, C, H, W) or (B, C, D, H, W) with one-hot encoding.
        smooth : float, default=1
            Smoothing constant to avoid division by zero.
            
        Returns
        -------
        Tensor
            Dice loss value (1 - Dice coefficient).
        """
        num_classes = inputs.shape[1]
        
        if num_classes == 1:
            # Binary segmentation
            inputs = F.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)
            
            intersection = (inputs * targets).sum()
            dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
            return 1 - dice
        else:
            # Multi-class segmentation
            inputs = F.softmax(inputs, dim=1)
            
            # Convert targets to one-hot if necessary
            if targets.ndim == inputs.ndim - 1:
                targets = F.one_hot(targets.long(), num_classes=num_classes)
                # Move channel dimension to position 1: (B, H, W, C) -> (B, C, H, W)
                targets = targets.permute(0, -1, *range(1, targets.ndim - 1)).float()
            
            # Flatten spatial dimensions: (B, C, H, W) -> (B, C, H*W)
            inputs = inputs.flatten(2)
            targets = targets.flatten(2)
            
            # Compute Dice per class
            intersection = (inputs * targets).sum(dim=2)  # (B, C)
            union = inputs.sum(dim=2) + targets.sum(dim=2)  # (B, C)
            dice_per_class = (2. * intersection + smooth) / (union + smooth)  # (B, C)
            
            # Select classes to include
            start_idx = 0 if self.include_background else 1
            dice_per_class = dice_per_class[:, start_idx:]
            
            # Apply weights if provided
            if self.weight is not None:
                weight = self.weight[start_idx:].to(dice_per_class.device)
                dice_per_class = dice_per_class * weight
            
            # Average across classes and batch
            dice = dice_per_class.mean()
            return 1 - dice


class DiceCELoss(Module):
    """Combined Dice and Cross-Entropy loss for binary and multi-class segmentation.
    
    For binary segmentation (inputs with 1 channel), uses BCE + Dice.
    For multi-class segmentation (inputs with >1 channels), uses CE + Dice.
    
    Parameters
    ----------
    weight : Tensor, optional
        A manual rescaling weight given to each class for both losses.
    include_background : bool, default=True
        Whether to include the background class in the Dice loss calculation.
    ce_weight : float, default=1.0
        Weight for the cross-entropy component.
    dice_weight : float, default=1.0
        Weight for the Dice loss component.
    """
    
    def __init__(self, weight=None, include_background=True, ce_weight=1.0, dice_weight=1.0):
        super(DiceCELoss, self).__init__()
        self.dice_loss = DiceLoss(weight=weight, include_background=include_background)
        self.weight = weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets, smooth=1):
        """Compute combined Dice and Cross-Entropy loss.
        
        Parameters
        ----------
        inputs : Tensor
            Predicted logits of shape (B, C, H, W) or (B, C, D, H, W)
            where C is the number of classes (C=1 for binary).
        targets : Tensor
            Ground truth of shape (B, H, W) or (B, D, H, W) with class indices,
            or (B, C, H, W) or (B, C, D, H, W) with one-hot encoding for binary case.
        smooth : float, default=1
            Smoothing constant for Dice loss.
            
        Returns
        -------
        Tensor
            Combined loss value.
        """
        num_classes = inputs.shape[1]
        
        # Compute Dice loss
        dice_loss = self.dice_loss(inputs, targets, smooth=smooth)
        
        if num_classes == 1:
            # Binary segmentation: use BCE
            inputs_sigmoid = F.sigmoid(inputs)
            ce_loss = F.binary_cross_entropy(inputs_sigmoid, targets, weight=self.weight, reduction='mean')
        else:
            # Multi-class segmentation: use CE
            # Ensure targets are class indices (not one-hot)
            if targets.ndim == inputs.ndim:
                # One-hot encoded targets -> class indices
                targets = torch.argmax(targets, dim=1)
            
            ce_loss = F.cross_entropy(inputs, targets.long(), weight=self.weight, reduction='mean')
        
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_loss(loss: str) -> Callable:
    """Get loss function by name.

    Parameters
    ----------
    loss : str
        Name of the loss function. Supported: "dice", "ce", "dicece".

    Returns
    -------
    Callable
        Corresponding loss function.
    """
    if loss == "dice":
        return DiceLoss()
    elif loss == "ce":
        return CrossEntropyLoss()
    elif loss == "dicece":
        return DiceCELoss()
    else:
        raise ValueError(f"Unsupported loss function: {loss}")