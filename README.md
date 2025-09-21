# 双图像分类任务项目文档

## 项目概述
本项目是一个基于PyTorch的深度学习项目，主要用于处理双图像输入的分类任务（推测为医学眼底图像分类）。项目包含数据处理、模型构建、训练、评估等完整流程，支持多种基础模型（ResNet、VGG、Xception、DenseNet等），并通过数据增强、多阶段训练、交叉验证等策略优化模型性能。


## 文件结构与功能说明

| 文件路径 | 主要功能 |
|---------|---------|
| `datastrength.py` | 数据集处理工具，负责复制图像数据并生成标签文件（`labels.txt`），从数据集中提取左右眼图像对及其对应标签 |
| `test_one_by_one.py` | 模型测试与评估脚本，加载训练好的模型，对测试集进行推理，计算并保存各标签的准确率、精确率、召回率、F1分数等指标 |
| `MLdecoder.py` | 提供ML解码器（MLDecoder）集成功能，可替换模型的分类头，支持ResNet、TResNet等模型 |
| `model.py` | 定义基础主干网络，基于预训练ResNet50，移除全连接层作为特征提取器 |
| `fifthtryfortrain..py` | 训练脚本，实现模型训练流程，支持训练/验证损失计算、学习率调度、模型保存（基于F1分数） |
| `tenthtryfortrain.py` | 训练主函数，使用ResNet50构建多尺度特征提取器，配置数据增强、优化器（Adam）和学习率调度器 |
| `xception3.py` | 基于Xception模型的训练脚本，使用5折交叉验证，自定义损失函数（conditional_mutual_exclusion_loss） |
| `seventhtrryfortraiin.py` / `DualImagevgg16.py` | 基于VGG16的训练脚本，使用FocalLoss损失函数 |
| `ninethtryfortrain.py` | 带L1正则化的ResNet50训练脚本 |
| `eighthtryfortrain.py` | 使用EQL损失函数的ResNet50训练脚本 |
| `eleventhtryfortrain.py` | 两阶段训练脚本：先在未过采样数据上训练全模型，再冻结特征提取器在过采样数据上训练分类头 |
| `xceptiontry.py` | 基于ResNet50的5折交叉验证训练脚本，合并多折结果计算总体指标 |
| `mutle_scale.py` / `mutle_scale(1).py` | 基于ResNet50多尺度特征提取器的训练脚本，对比不同优化器（Adam/SGD）效果 |
| `densenettry.py` | 基于DenseNet161的5折交叉验证训练脚本 |


## 核心功能模块

### 1. 数据处理
- **数据集构建**：通过`DualImageDataset`加载双图像对（如左右眼图像），支持训练/验证集划分
- **数据增强**：训练集使用随机裁剪、垂直翻转、仿射变换（旋转、平移、缩放）、高斯模糊等增强策略，验证集仅做 resize 和标准化
- **标签生成**：通过`copy_dataset`函数从原始数据中提取图像对及其标签，保存为`labels.txt`


### 2. 模型构建
- **基础主干网络**：支持ResNet50、VGG16、Xception41、DenseNet161等预训练模型，作为特征提取器
- **多尺度特征提取**：通过`MultiScaleResNet`/`MultiScaleDenseNet`提取多尺度特征，增强特征表达能力
- **双图像处理**：通过`DualImageResNet`处理双输入图像（如左右眼），融合两者特征进行分类
- **分类头**：可替换为MLDecoder（`MLdecoder.py`），或使用默认全连接层


### 3. 训练过程
- **损失函数**：支持MSELoss、FocalLoss、EQL、conditional_mutual_exclusion_loss等，适应不同分类场景
- **优化器与调度器**：主要使用Adam优化器（带L2正则化），配合ReduceLROnPlateau学习率调度器（根据验证损失调整学习率）
- **训练策略**：
  - 多阶段训练：先训练全模型，再冻结特征提取器微调分类头
  - 交叉验证：5折交叉验证（`xception3.py`/`xceptiontry.py`/`densenettry.py`），减少过拟合
  - 过采样：针对不平衡数据，使用过采样数据集优化少数类性能（`eleventhtryfortrain.py`）


### 4. 评估方法
- **指标计算**：计算各标签的准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1分数，以及总体指标（平均精确率、平均召回率、综合得分等）
- **结果保存**：测试结果（图像对、预测输出、真实标签）保存至`predictions.txt`，包含详细指标


## 使用说明
1. **数据准备**：通过`datastrength.py`的`copy_dataset`函数处理原始数据，生成训练/验证数据集及标签文件
2. **模型训练**：选择对应模型的训练脚本（如`tenthtryfortrain.py`for ResNet50，`DualImagevgg16.py`for VGG16），配置数据路径和超参数（batch_size、epoch、学习率等），运行训练
3. **模型评估**：使用`test_one_by_one.py`的`test_model`函数加载训练好的模型，对测试集进行评估，生成详细结果文件


## 备注
- 代码中针对医学图像特点（如类别互斥性），在推理阶段加入条件调整（如`preds[condition3, 0] = 1`），优化预测逻辑
- 支持多设备运行（自动检测GPU/CPU），通过`device`参数控制
- 模型训练过程中使用`tqdm`显示进度条，便于监控训练状态
