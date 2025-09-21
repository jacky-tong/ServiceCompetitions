from torchvision.models import resnet50
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # 使用预训练的ResNet50作为主干网络
        self.resnet50 = resnet50(pretrained=True)
        # 根据需要调整最后一层，例如去掉全连接层
        self.resnet50.fc = nn.Identity()

    def forward(self, x):
        return self.resnet50(x)