import os
import random
import shutil
import uuid
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from collections import defaultdict
import numpy as np
import cv2

# 固定随机种子保证可重复性
random.seed(42)

# 路径配置
IMAGE_DIR = r"D:\Serviceoutsourcingcompetition\newpackage\mine\preall"
TRAIN_DIR = r"D:\Serviceoutsourcingcompetition\newpackage\mine\pretrain2"
VAL_DIR = r"D:\Serviceoutsourcingcompetition\newpackage\mine\preval2"
TEST_DIR = r"D:\Serviceoutsourcingcompetition\newpackage\mine\test"
ANNOTATION_FILE = r"D:\Serviceoutsourcingcompetition\newpackage\Training Set\Annotation\training annotation (English).xlsx"
# 疾病标签列
LABEL_COLUMNS = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# 定向过采样配置（支持小数倍数）
TARGET_MULTIPLIERS = {
    'G': 4,
    'C': 4,
    'A': 5,
    'H': 10,
    'M': 3,
    'N': 1,
    'D': 1.2,
    'O': 1.3
}

MAX_AUGMENT_PER_SAMPLE = 2  # 每个样本最多增强次数

class MedicalImageAugmentor:
    """医学图像增强优化版（已移除弹性形变）"""
    def __init__(self):
        # 保守的增强参数配置
        self.params = {
            'rotation': (-10, 10),         # 旋转角度范围
            'scale': (0.92, 1.08),       # 缩放比例范围  
            'translation': (-0.1, 0.1), # 平移比例范围
            'brightness': (0.95, 1.05),    # 亮度调整范围
            'contrast': (0.97, 1.03),      # 对比度调整范围
            'gaussian_noise': 0.05       # 高斯噪声比例
        }

    def add_gaussian_noise(self, image):
        """添加高斯噪声"""
        array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, 25 * self.params['gaussian_noise'], array.shape)
        noisy = np.clip(array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    
    def augment_pair(self, left_path, right_path):
        """同步增强左右眼图像对"""
        # 加载图像
        left = Image.open(left_path)
        right = Image.open(right_path)

        # 生成全局增强参数
        angle = np.random.uniform(*self.params['rotation'])
        scale = np.random.uniform(*self.params['scale'])
        tx = int(np.random.uniform(*self.params['translation']) * left.width)
        ty = int(np.random.uniform(*self.params['translation']) * left.height)

        # 定义同步增强流程
        def sync_augment(img):
            # 几何变换
            img = img.rotate(angle, resample=Image.BILINEAR, expand=True)
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
            img = ImageOps.expand(img, border=(tx, ty, -tx, -ty), fill=0)
            
            # 颜色调整（概率性应用）
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(np.random.uniform(*self.params['brightness']))
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(np.random.uniform(*self.params['contrast']))
                
            # 添加噪声（概率性应用） 
            if random.random() < 0.2:
                img = self.add_gaussian_noise(img)
                
            # 边缘保留模糊（概率性应用）
            if random.random() < 0.3:
                img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
                
            return img

        return sync_augment(left), sync_augment(right)

class TargetedOversampler:
    """定向多标签过采样器（支持小数倍数）"""
    def __init__(self, annotation_df):
        self.df = annotation_df
        self.label_stats = defaultdict(int)
        self.label_needs = defaultdict(int)
        self.sample_augment_count = defaultdict(int)
        self.augmentor = MedicalImageAugmentor()
        self.total_generated = defaultdict(int)

    def analyze_distribution(self, train_samples):
        """分析初始分布并计算需求（使用四舍五入）"""
        for sample in train_samples:
            record = self.df[self.df['Left-Fundus'] == sample].iloc[0]
            for label in LABEL_COLUMNS:
                if record[label] == 1:
                    self.label_stats[label] += 1

        print("\n[初始分布]")
        for label in LABEL_COLUMNS:
            original = self.label_stats[label]
            target = round(original * TARGET_MULTIPLIERS[label])  # 四舍五入
            need = max(0, target - original)
            self.label_needs[label] = need
            print(f"{label}: {original} -> 目标 {target} (需要生成 {need})")

    def generate_augmentations(self, train_left_samples):
        """改进后的过采样逻辑"""
        print("\n[开始过采样]")
        augmentation_records = []
        new_samples = []

        while any(self.label_needs.values()):
            # 动态计算所有候选样本的优先级
            candidates = []
            for sample in train_left_samples:
                # 跳过已达增强上限的样本
                if self.sample_augment_count.get(sample, 0) >= MAX_AUGMENT_PER_SAMPLE:
                    continue
                
                # 动态计算当前有效标签
                record = self.df[self.df['Left-Fundus'] == sample].iloc[0]
                valid_labels = [label for label in LABEL_COLUMNS 
                               if record[label] == 1 and self.label_needs.get(label, 0) > 0]
                
                if valid_labels:
                    candidates.append((sample, len(valid_labels)))

            if not candidates:
                print("\n警告：无法满足所有需求，剩余需求:")
                for label, need in self.label_needs.items():
                    if need > 0:
                        print(f"{label}: {need}")
                break

            # 选择优先级最高的样本
            best_sample, max_priority = max(candidates, key=lambda x: x[1])

            # 执行增强操作
            unique_id = uuid.uuid4().hex[:8]
            new_left = f"{os.path.splitext(best_sample)[0]}_aug_{unique_id}_left.jpg"
            new_right = new_left.replace("_left.jpg", "_right.jpg")

            left_img, right_img = self.augmentor.augment_pair(
                os.path.join(IMAGE_DIR, best_sample),
                os.path.join(IMAGE_DIR, best_sample.replace("_left.jpg", "_right.jpg"))
            )
            left_img.save(os.path.join(TRAIN_DIR, new_left))
            right_img.save(os.path.join(TRAIN_DIR, new_right))

            new_samples.append((new_left, new_right))
            record = self.df[self.df['Left-Fundus'] == best_sample].iloc[0].copy()
            record['Left-Fundus'] = new_left
            record['Right-Fundus'] = new_right
            augmentation_records.append(record)

            # 获取样本真实标签并更新需求
            record = self.df[self.df['Left-Fundus'] == best_sample].iloc[0]
            true_labels = [label for label in LABEL_COLUMNS if record[label] == 1]
            
            updated_labels = []
            for label in true_labels:
                if self.label_needs.get(label, 0) > 0:
                    self.label_needs[label] -= 1
                    self.total_generated[label] += 1
                    updated_labels.append(label)
            
            print(f"生成样本 {new_left}，减少需求: {updated_labels}")
            print("当前剩余需求:", {k:v for k,v in self.label_needs.items() if v>0})

            # 更新增强计数
            self.sample_augment_count[best_sample] = self.sample_augment_count.get(best_sample, 0) + 1

        # 更新DataFrame
        if augmentation_records:
            self.df = pd.concat([self.df, pd.DataFrame(augmentation_records)], ignore_index=True)
        
        return new_samples

def prepare_directories():
    """初始化存储目录"""
    for path in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(path, exist_ok=True)
        if os.path.exists(os.path.join(path, "labels.txt")):
            os.remove(os.path.join(path, "labels.txt"))

def copy_dataset(samples, target_dir):
    """复制数据集并生成标注"""
    with open(os.path.join(target_dir, "labels.txt"), "a") as f:
        for left, right in set(samples):
            for img in [left, right]:
                src = os.path.join(IMAGE_DIR, img)
                dst = os.path.join(target_dir, img)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)
            
            record = df[(df['Left-Fundus'] == left) & (df['Right-Fundus'] == right)].iloc[0]
            labels = ','.join([str(record[col]) for col in LABEL_COLUMNS])
            f.write(f"{left},{right},{labels}\n")

if __name__ == "__main__":
    prepare_directories()
    df = pd.read_excel(ANNOTATION_FILE)
    
    all_samples = [f for f in os.listdir(IMAGE_DIR) if f.endswith("_left.jpg")]
    random.shuffle(all_samples)
    
    # 精确计算数据集划分（四舍五入）
    total = len(all_samples)
    train_size = round(total * 0.8)
    train_left_samples = all_samples[:train_size]
    train_samples = [(f, f.replace("_left.jpg", "_right.jpg")) for f in train_left_samples]
    
    sampler = TargetedOversampler(df)
    sampler.analyze_distribution(train_left_samples)
    augmented_samples = sampler.generate_augmentations(train_left_samples)
    
    final_train_samples = train_samples + augmented_samples
    random.shuffle(final_train_samples)
    
    df = sampler.df
    copy_dataset(final_train_samples, TRAIN_DIR)
    copy_dataset([(f, f.replace("_left.jpg", "_right.jpg")) for f in all_samples[train_size:]], VAL_DIR)

    print("\n[最终统计]")
    print("初始目标需求:")
    for label in LABEL_COLUMNS:
        original = sampler.label_stats[label]
        print(f"{label}: {original} -> 目标 {round(original*TARGET_MULTIPLIERS[label])}")

    print("\n实际生成数量:")
    for label in LABEL_COLUMNS:
        final_count = sampler.label_stats[label] + sampler.total_generated.get(label, 0)
        print(f"{label}: {final_count} (生成 {sampler.total_generated.get(label,0)})")

    print(f"\n训练集: {len(final_train_samples)} 样本对")
    print(f"验证集: {len(all_samples[train_size:])} 样本对")