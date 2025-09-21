import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score
import os
from PIL import Image

# 假设 DualImageResNet 的定义在其他文件中
from xceptiontry import DualImageResNet, MultiScaleResNet

class TestDataset(Dataset):
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

def test_model(model, test_loader, device, model_path="best_model.pth"):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_outputs = []

    # 打开文件以写入预测结果
    with open("predictions.txt", "w") as f:
        with torch.no_grad():
            for img1, img2, label in test_loader:  # 每次加载一个样本
                img1, img2 = img1.to(device), img2.to(device)
                
                # 直接推理单个样本
                outputs = model(img1, img2)
                
                # 转换为预测结果
                preds = (outputs > 0.5).float()
                
                # 应用条件调整
                #if (outputs[0, 1:8] < 0.5).all():
                    #preds[0, 0] = 1
                
                formatted_output = np.array2string(
                    outputs.squeeze().cpu().numpy(),
                    precision=4,       # 保留 3 位小数
                    suppress_small=True # 抑制科学计数法显示小数值
                )

                # 将预测结果写入文件
                f.write(f"Image Pair: {test_loader.dataset.image_pairs[len(all_labels)]}\n")
                f.write(f"Predicted Outputs: {formatted_output}\n")
                f.write(f"Predicted Labels: {preds.squeeze().cpu().numpy()}\n")
                f.write(f"True Labels: {label.squeeze().cpu().numpy()}\n")
                f.write("-" * 50 + "\n")
                
                all_preds.append(preds.squeeze().cpu().numpy())
                all_labels.append(label.squeeze().cpu().numpy())
                all_outputs.append(outputs.squeeze().cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_outputs = np.array(all_outputs)

    overall_precision = 0.0
    overall_recall = 0.0

    # 各标签指标
    for i in range(model.num_classes):
        accuracy = accuracy_score(all_labels[:, i], all_preds[:, i])
        precision = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        overall_precision += precision
        recall = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        overall_recall += recall
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        print(f"Label {i + 1}: Acc={accuracy:.4f} Prec={precision:.4f} Rec={recall:.4f} F1={f1:.4f}")

    overall_precision /= model.num_classes
    overall_recall /= model.num_classes

    all_labels = all_labels.flatten()
    all_preds = all_preds.flatten()
    overall_accuracy = accuracy_score(all_labels, all_preds)

    final_score = (overall_accuracy + overall_precision + overall_recall)/3
    
    print(f"\nAccuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall final_score: {final_score:.4f}")

    # 将总体指标写入文件
    with open("predictions.txt", "a") as f:
        f.write("\nOverall Metrics:\n")
        f.write(f"Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"Overall Precision: {overall_precision:.4f}\n")
        f.write(f"Overall Recall: {overall_recall:.4f}\n")
        f.write(f"Overall final_score: {final_score:.4f}\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    transform = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(
        data_dir=r"E:\Pred_Val_Dataset",
        transform=transform
    )
    
    # 关键修改：设置 batch_size=1 实现完全逐样本加载
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 强制每次加载一个样本
        shuffle=False,
        num_workers=0   # 为避免潜在问题，num_workers设为0
    )
    '''
    base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    base_resnet = nn.Sequential(*list(base_resnet.children())[:-2])
    model = DualImageResNet(base_model=base_resnet)
    '''
    base_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    base_model = MultiScaleResNet(base_resnet)  # 使用多尺度特征提取器
    model = DualImageResNet(base_model=base_model).to(device)


    test_model(model, test_loader, device, model_path=r"best_model_fold_2.pth")

if __name__ == "__main__":
    main()