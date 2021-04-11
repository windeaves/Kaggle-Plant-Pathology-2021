from torchvision import models
from torch import nn
import torch

resnet50_cls = models.resnet50(pretrained=True)

class AvgPool(nn.Module):
    
    def forward(self, x):
        return nn.functional.avg_pool2d(x, x.shape[2:])

class ResNet50(nn.Module):
    
    def __init__(self, num_outputs, dropout = 0.5):
        super(ResNet50, self).__init__()
        self.resnet = resnet50_cls
        self.resnet.layer4 = nn.Sequential(
            nn.Dropout(dropout),
            resnet50_cls.layer4
        )
        
        self.resnet.avgpool = AvgPool()
        self.resent.fc = nn.Linear(2048, num_outputs)

        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.resnet(x)

class MultiLabelCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(MultiLabelCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        output = (1 - 2 * target) * output
        output_neg = output - target * 1e12
        output_pos = output - (1 - target) * 1e12
        zeros = torch.zeros_like(output[..., :1])
        output_neg = torch.cat([output_neg, zeros], dim=-1)
        output_pos = torch.cat([output_pos, zeros], dim=-1)
        loss_neg = torch.logsumexp(output_neg, dim=-1)
        loss_pos = torch.logsumexp(output_pos, dim=-1)
        return torch.sum(loss_neg + loss_pos) / len(output)