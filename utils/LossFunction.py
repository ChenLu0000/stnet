import torch 
import torch.nn as nn
import torch.nn.functional as F

def flatten(tensor):
    C = tensor.size(1)
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))     
    transposed = tensor.permute(axis_order)
    return transposed.contiguous().view(C, -1)

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    def forward(self, output, target):
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target)

        intersect = (output * target).sum(-1) + self.epsilon
        denominator = (output + target).sum(-1) + self.epsilon
        dice = intersect / denominator
        dice = torch.mean(dice)
        dice_loss = 1 - dice

        return dice_loss

class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        bce_loss = self.bce_with_logits_loss(logits, targets)
        return bce_loss
        
class BCEDiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce_loss = BCELoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, logits, targets):
        targets_one_hot = F.one_hot(targets, num_classes=logits.shape[1]).permute(0,3,1,2).float()
        loss = self.bce_loss(logits, targets_one_hot) + self.dice_loss(logits, targets_one_hot)
        return loss