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

class EQL(nn.Module):
    def __init__(self, gamma=12, mu=0.8, alpha=4.0):
        super(EQL, self).__init__()
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

    def forward(self, logits, targets):
        pos_mask = targets > 0
        neg_mask = targets == 0

        # 正样本损失
        pos_logits = logits[pos_mask]
        if pos_logits.numel() == 0:
            pos_loss = torch.tensor(0.0, device=logits.device)
        else:
            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))

        # 负样本损失
        neg_logits = logits[neg_mask]
        if neg_logits.numel() == 0:
            neg_loss = torch.tensor(0.0, device=logits.device)
        else:
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            # 均衡负样本损失
            neg_prob = neg_logits.sigmoid()
            easy_neg_index = (neg_prob > self.mu).nonzero(as_tuple=True)[0]
            hard_neg_index = (neg_prob <= self.mu).nonzero(as_tuple=True)[0]
            if easy_neg_index.numel() > 0:
                neg_loss = neg_loss.clone()  # 避免原位操作警告
                neg_loss[easy_neg_index] = neg_loss[easy_neg_index] * (torch.pow(neg_prob[easy_neg_index], self.gamma) * self.alpha)

        # 合并正负样本损失
        num_pos = pos_mask.sum().item()
        num_neg = neg_mask.sum().item()
        if num_pos + num_neg == 0:
            return torch.tensor(0.0, device=logits.device)
        loss = (pos_loss.sum() + neg_loss.sum()) / (num_pos + num_neg)

        return loss


class FocalClassBalancedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, beta=0.9999):
        super(FocalClassBalancedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, inputs, targets):
        # 计算类别平衡因子
        num_classes = inputs.size(-1)
        num_samples_per_class = [(targets == i).sum().item() for i in range(num_classes)]
        effective_num = []
        for num in num_samples_per_class:
            if num == 0:
                effective_num.append(1e-6)  # 避免除零错误
            else:
                effective_num.append(1.0 - torch.pow(torch.tensor(self.beta, device=inputs.device), torch.tensor(num, device=inputs.device)))
        effective_num = torch.tensor(effective_num, device=inputs.device)
        cb_weights = (1.0 - self.beta) / effective_num
        cb_weights = cb_weights / cb_weights.sum() * num_classes
        cb_weights = cb_weights.to(inputs.device)

        # 计算焦点损失
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = cb_weights * self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()



class GHMC(nn.Module):
    def __init__(self, bins=10, momentum=0):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins + 1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0] * bins

    def forward(self, pred, target):
        # 计算梯度模长
        g = torch.abs(pred.sigmoid().detach() - target)

        # 计算每个区间的样本数量
        tot = target.numel()
        n = 0
        weights = torch.zeros_like(pred)
        for i in range(self.bins):
            inds = (g >= self.edges[i]) & (g < self.edges[i + 1])
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if self.momentum > 0:
                    self.acc_sum[i] = self.momentum * self.acc_sum[i] + (1 - self.momentum) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(pred, target, weight=weights, reduction='sum') / tot
        return loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        log_softmax = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)

        # 应用标签平滑
        target_smooth = targets * (1 - self.smoothing) + self.smoothing / n_classes

        loss = -(target_smooth * log_softmax).sum(dim=-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
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

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets, smooth=1e-5):
        inputs = torch.sigmoid(inputs)
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        return 1 - tversky
        
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


# 模型结构
class DualImageResNet(nn.Module):
    def __init__(self, base_model, num_classes=8):
        super(DualImageResNet, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # 冻结基础模型的前五层
        children = list(self.base_model.children())
        for child in children[:5]:  # 取前五个子模块
            for param in child.parameters():
                param.requires_grad = False  # 冻结参数
        
        self.classifier = nn.Sequential(
            nn.Linear(2048*2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, img1, img2):
        feat1 = self.base_model(img1)
        feat2 = self.base_model(img2)
        
        feat1 = feat1.view(feat1.size(0), -1)
        feat2 = feat2.view(feat2.size(0), -1)
        
        fused_feat = torch.cat((feat1, feat2), dim=1)
        
        output = self.classifier(fused_feat)
        return output


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, save_path="model.pth"):
    model.to(device)
    best_val_loss = float("inf")
    best_f1_score = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        # 使用tqdm显示训练进度条
        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
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

        # 总体指标
        overall_accuracy = np.mean(np.all(all_labels == all_preds, axis=1))
        overall_precision = precision_score(all_labels, all_preds, average='micro', zero_division=1)
        overall_recall = recall_score(all_labels, all_preds, average='micro', zero_division=1)
        overall_f1 = f1_score(all_labels, all_preds, average='micro')
        overall_auc = roc_auc_score(all_labels, all_outputs)
        
        print(f"Strict Accuracy: {overall_accuracy:.4f}")
        print(f"overall Precision: {overall_precision:.4f}")
        print(f"overall Recall: {overall_recall:.4f}")
        print(f"overall F1: {overall_f1:.4f}")
        print(f"overall auc: {overall_auc:.4f}")

        # 保存最佳模型
        if overall_f1 > best_f1_score:
            best_f1_score = overall_f1
            torch.save(model.state_dict(), save_path)
            print(f"\nBest model saved with f1_score : {best_f1_score:.4f}")

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

    train_dataset = DualImageDataset(data_dir=r"D:\Serviceoutsourcingcompetition\newpackage\mine\pretrain2", transform=transform["train"])
    val_dataset = DualImageDataset(data_dir=r"D:\Serviceoutsourcingcompetition\newpackage\mine\preval2", transform=transform["val"])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    base_resnet = nn.Sequential(*list(base_resnet.children())[:-1])
    model = DualImageResNet(base_model=base_resnet)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion =EQL()
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
        save_path="best_model.pth"
    )

if __name__ == "__main__":
    main()