import torch.nn.functional as F 

class DiceLoss:
    """
    From:
    https://medium.com/data-scientists-diary/implementation-of-dice-loss-vision-pytorch-7eef1e438f68
    Modified to handle multi-class targets and match input dimensions.

    Simple Dice Loss algorithm for multiple labels
    """
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)

        targets = targets.long()
        if targets.dim() == 4:
            targets = targets[:, 0, :, :]
        elif targets.dim() != 3:
            raise ValueError(f"Unsupported target shape for DiceLoss: {targets.shape}. Expected [N, 1, H, W] or [N, H, W].")

        targets = F.one_hot(targets, num_classes=6) 

        targets = targets.permute(0, 3, 1, 2).float()

        intersection = (inputs * targets).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice.mean()