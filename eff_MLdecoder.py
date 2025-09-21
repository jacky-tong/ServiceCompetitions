import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,roc_auc_score, cohen_kappa_score
import os
from PIL import Image
import matplotlib.pyplot as plt
from MLdecoder import MLDecoder
        
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
class DualImageEfficientNet(nn.Module):
    def __init__(self, base_model, num_classes=8):
        super(DualImageEfficientNet, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # ML-Decoder
        self.ml_decoder = MLDecoder(num_classes,initial_num_features=3584)

    def forward(self, img1, img2):
        feat1 = self.base_model(img1)
        feat2 = self.base_model(img2)

        # Concatenate features
        fused_feat = torch.cat((feat1, feat2), dim=1)

        # ML-Decoder
        output = self.ml_decoder(fused_feat)
        output = torch.sigmoid(output)
        return output 
    
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=10, save_path="model.pth"):
    model.to(device)
    best_val_loss = float("inf")
    best_acc = 0.0

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用tqdm显示训练进度条
        for img1, img2, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(img1, img2)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
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

                correct += (preds == labels).sum().item()
                total += labels.numel()

        val_loss /= len(val_loader)
        val_accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        # 更新学习率
        scheduler.step(val_loss)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        all_labels = np.concatenate(all_labels, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_outputs = np.concatenate(all_outputs, axis=0)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}]:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_accuracy:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.2e}")

        # 各标签指标
        for i in range(model.num_classes):
            accuracy = accuracy_score(all_labels[:, i], all_preds[:, i])
            precision = precision_score(all_labels[:, i], all_preds[:, i], zero_division=1)
            recall = recall_score(all_labels[:, i], all_preds[:, i], zero_division=1)
            f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=1)
            print(f"Label {i + 1}: Acc={accuracy:.4f} Prec={precision:.4f} Rec={recall:.4f} F1={f1:.4f}")

        overall_precision = precision_score(all_labels, all_preds, average='micro', zero_division=1)
        overall_recall = recall_score(all_labels, all_preds, average='micro', zero_division=1)
        overall_f1 = f1_score(all_labels, all_preds, average='micro')

        all_labels = all_labels.flatten()
        all_preds = all_preds.flatten()
        all_outputs = all_outputs.flatten()

        # 总体指标
        overall_accuracy = accuracy_score(all_labels, all_preds)
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

        # 绘制并保存训练和验证的损失和准确率曲线
        os.makedirs("training_plots", exist_ok=True)
        epochs = range(1, epoch + 2)

        plt.figure()
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.savefig(f'training_plots/loss_curve.png')
        plt.close()

        plt.figure()
        plt.plot(epochs, train_accuracies, label='Train Accuracy')
        plt.plot(epochs, val_accuracies, label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        plt.savefig(f'training_plots/accuracy_curve.png')
        plt.close()

    print("\nTraining complete.")

# 主函数
def main():
    transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(280),  # 随机裁剪
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
            transforms.Resize((224,224)),
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

    # 使用 EfficientNet-B4 作为主干网络
    base_model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT).features
    model = DualImageEfficientNet(base_model=base_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
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
        num_epochs=40, 
        save_path="best_model.pth"
    )

if __name__ == "__main__":
    main()