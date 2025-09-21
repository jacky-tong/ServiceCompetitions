import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,roc_auc_score, cohen_kappa_score
import os
from PIL import Image
        
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算基础交叉熵损失
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # 计算概率因子pt（与原始公式等价）
        pt = torch.exp(-BCE_loss)  # pt = p when y=1, pt = 1-p when y=0
        
        # 动态生成alpha系数张量
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 组合Focal Loss
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        # 根据reduction模式返回结果
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Dataset类
class DualImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = []
        self.labels = []

        label_file = os.path.join(data_dir, "labels.txt")
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                img1_path = os.path.join(data_dir, parts[0])
                img2_path = os.path.join(data_dir, parts[1])
                labels = list(map(int, parts[2:]))
                self.image_pairs.append((img1_path, img2_path))
                self.labels.append(labels)

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        img2 = transforms.functional.hflip(img2)  # 直接操作 PIL 图像

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MultiScaleResNet(nn.Module):
    def __init__(self, resnet_model):
        super(MultiScaleResNet, self).__init__()
        # 分解ResNet的各层
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 

        # 添加SE模块
        self.se1 = SEBlock(256)  # layer1输出通道数
        self.se2 = SEBlock(512)  # layer2输出通道数
        self.se3 = SEBlock(1024) # layer3输出通道数
        self.se4 = SEBlock(2048) # layer4输出通道数

    def forward(self, x):
        # 前向传播并收集多尺度特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.se1(x)  # 添加SE模块
        l2 = self.layer2(x)  # 512通道特征
        l2 = self.se2(l2)  # 添加SE模块
        l3 = self.layer3(l2)  # 1024通道特征
        l3 = self.se3(l3)  # 添加SE模块
        l4 = self.layer4(l3)  # 2048通道特征
        l4 = self.se4(l4)  # 添加SE模块

        # 对每个层次进行全局池化
        feat2 = self.avgpool(l2).flatten(1)
        feat3 = self.avgpool(l3).flatten(1)
        feat4 = self.avgpool(l4).flatten(1)

        return feat2, feat3, feat4

class DualImageResNet(nn.Module):
    def __init__(self, base_model, num_classes=8):
        super(DualImageResNet, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # 冻结前五层（根据ResNet的实际结构）
        children = list(self.base_model.children())
        for child in children[:5]:  # conv1, bn1, relu, maxpool, layer1
            for param in child.parameters():
                param.requires_grad = False
        
        # 调整分类器输入维度（512+1024+2048）*2 = 7168
        self.classifier = nn.Sequential(
            nn.Linear(7168, 1024),  # 增大中间层维度
            nn.ReLU(),
            nn.Dropout(0.4),        # 增加dropout防止过拟合
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, img1, img2):
        # 获取双图像的多尺度特征
        f2_1, f3_1, f4_1 = self.base_model(img1)
        f2_2, f3_2, f4_2 = self.base_model(img2)
        
        # 特征拼接
        fused_feat = torch.cat([
            f2_1, f3_1, f4_1,
            f2_2, f3_2, f4_2
        ], dim=1)
        
        return self.classifier(fused_feat)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, save_path="model.pth", l1_lambda=0.001):
    model.to(device)
    best_val_loss = float("inf")
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 使用tqdm显示训练进度条
        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)

            # 添加 L1 正则化项
            l1_reg = torch.tensor(0., requires_grad=True).to(device)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_reg = l1_reg + torch.norm(param, 1)
            loss = loss + l1_lambda * l1_reg

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_outputs = []

        # 使用tqdm显示验证进度条
        with torch.no_grad():
            for img1, img2, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False):
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                outputs = model(img1, img2)

                val_loss += criterion(outputs, labels).item()
                preds = (outputs > 0.5).float()

                condition3 = (outputs[:, 1:8] < 0.5).all(dim=1)
                preds[condition3, 0] = 1

                all_outputs.append(outputs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss /= len(val_loader)
        # 更新学习率
        scheduler.step(val_loss)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']


        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")

        # 各标签指标
        for i in range(model.num_classes):
            accuracy = accuracy_score(all_labels[:, i], all_preds[:, i])
            precision = precision_score(all_labels[:, i], all_preds[:, i], zero_division=1)
            recall = recall_score(all_labels[:, i], all_preds[:, i], zero_division=1)
            f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=1)
            print(f"Label {i + 1}: Acc={accuracy:.4f} Prec={precision:.4f} Rec={recall:.4f} F1={f1:.4f}")

        all_labels = all_labels.flatten()
        all_preds = all_preds.flatten()
        all_outputs = all_outputs.flatten()

        # 总体指标
        overall_accuracy = accuracy_score(all_labels, all_preds)
        overall_precision = precision_score(all_labels, all_preds, average='micro', zero_division=1)
        overall_recall = recall_score(all_labels, all_preds, average='micro', zero_division=1)
        overall_f1 = f1_score(all_labels, all_preds, average='micro')
        overall_auc = roc_auc_score(all_labels, all_outputs)

        print(f"Accuracy: {overall_accuracy:.4f}")
        print(f"overall Precision: {overall_precision:.4f}")
        print(f"overall Recall: {overall_recall:.4f}")
        print(f"overall F1: {overall_f1:.4f}")
        print(f"overall auc: {overall_auc:.4f}")

        # 保存最佳模型
        if overall_accuracy > best_acc:
            best_acc = overall_accuracy
            torch.save(model.state_dict(), save_path)
            print(f"\nBest model saved with acc : {best_acc:.4f}")

    print("\nTraining complete.")

# 主函数
def main():
    transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(448),
            transforms.RandomVerticalFlip(p=0.3),  # 新增垂直翻转
            transforms.RandomAffine(
                degrees=10, 
                translate=(0.1, 0.1),  # 小幅平移
                scale=(0.9, 1.1)       # 小幅缩放
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(       # 高斯模糊
                    kernel_size=5, 
                    sigma=(0.1, 2.0))], 
                p=0.2
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    batch_size = 16
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])

    train_dataset = DualImageDataset(data_dir=r"D:\Serviceoutsourcingcompetition\newpackage\mine\pretrain", transform=transform["train"])
    val_dataset = DualImageDataset(data_dir=r"D:\Serviceoutsourcingcompetition\newpackage\mine\preval", transform=transform["val"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 初始化基础模型
    base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    base_model = MultiScaleResNet(base_resnet)  # 使用多尺度特征提取器
    model = DualImageResNet(base_model=base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.3,
        patience=2,
        min_lr=1e-6
    )

    train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer,
        scheduler,  # 传入调度器
        device, 
        num_epochs=30, 
        save_path="best_model.pth",
        l1_lambda=0.001
    )

if __name__ == "__main__":
    main()