# SDD-LSKD 融合：知识蒸馏增强方法

[English](README_en.md) | **中文版**

## 🔥 项目概述

本项目实现了一种新颖的融合方法，结合了**尺度解耦蒸馏 (Scale Decoupled Distillation, SDD)** 和**知识蒸馏中的 Logit 标准化 (Logit Standardization in Knowledge Distillation, LSKD)**，以实现教师和学生网络之间的增强知识传递。

本实现基于以下项目：
- **[SDD](https://github.com/shicaiwei123/SDD-CVPR2024)**: Scale Decoupled Distillation (CVPR 2024)
- **[LSKD](https://github.com/sunshangquan/logit-standardization-KD)**: Logit Standardization in Knowledge Distillation

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Compatible_CUDA_12.4-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## 🎯 核心特性

- **🔧 即插即用**：轻松集成到现有知识蒸馏框架
- **🚀 GPU 优化**：完整 CUDA 12.4 支持，高效多尺度处理
- **📊 全面兼容**：支持各种教师-学生网络架构
- **🎛️ 灵活配置**：针对不同场景的灵活超参数设置
- **📈 效果验证**：在 CIFAR-100 上验证，性能稳定提升

## 🧠 方法概述

### SDD (尺度解耦蒸馏)
- **核心思想**：将全局 logit 输出分解为多尺度局部输出
- **优势**：避免传递模糊的混合语义知识
- **特点**：
  - 多尺度池化进行细粒度知识提取
  - 一致性 vs 互补性知识分类
  - 针对困难样本的自适应权重调整

### LSKD (知识蒸馏中的 Logit 标准化)
- **核心思想**：Z-score 标准化聚焦于 logit 关系而非幅度匹配
- **优势**：减轻学生网络匹配教师 logit 幅度的压力
- **特点**：
  - Z-score 标准化：`(logit - mean) / std`
  - 使用加权标准差的自适应温度
  - 改进对相对知识模式的聚焦

### 融合策略
我们的 SDD-LSKD 融合方法将 logit 标准化应用于多尺度输出，结合了 SDD 和 LSKD 的优势：
- **基础方法**: 使用传统 KD (Knowledge Distillation) 作为知识传递基础
- **多尺度增强**: 通过 SDD 的多尺度分解进行细粒度知识传递
- **标准化优化**: 通过 LSKD 的标准化 logit 关系改进学习聚焦
- **广泛适用**: 在同构和异构教师-学生对上均表现出色

## 🛠️ 安装配置

### 环境要求
- **CUDA 12.4**：必需，依赖 CUDA 的其他库都需要兼容版本
- **Python 3.8+**
- **PyTorch 兼容 CUDA 12.4 版本**

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/Shardzzi/SDD-LSKD-Fusion.git
cd SDD-LSKD-Fusion

# 使用 conda 配置基础环境
conda env create -f sdd-lskd-fusion.yml
conda activate sdd-lskd-fusion

# 安装 CUDA 12.4 兼容的 PyTorch
# 注意：需要根据具体 CUDA 12.4 版本安装对应的 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# 使用 pip 安装其他依赖
pip install -r requirements.txt

# 验证 CUDA 可用性
python -c "import torch; print(f'CUDA 可用: {torch.cuda.is_available()}, CUDA 版本: {torch.version.cuda}')"
```

### 2. 下载预训练教师模型

```bash
# 下载预训练教师模型
bash fetch_pretrained_teachers.sh
```

### 3. 数据集准备

CIFAR-100 数据集将在首次运行时自动下载。其他数据集：

```bash
# ImageNet（可选）
# 从 https://image-net.org/ 下载并放置在 ./data/imagenet

# CUB-200（可选）
# 下载预训练模型并放置在 ./save/cub200/
```

## 🚀 快速开始

### 基础训练

```bash
# 快速测试（5 轮训练，推荐首次运行）
PYTHONPATH=. python train_origin.py --cfg configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml --gpu 0 --M "[1,2,4]"

# 完整多尺度蒸馏训练
PYTHONPATH=. python train_origin.py --cfg configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml --gpu 0 --M "[1,2,4]"

# 仅全局蒸馏（类似 LSKD 行为）
PYTHONPATH=. python train_origin.py --cfg configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml --gpu 0 --M "[1]"
```

### 批量测试

```bash
# 运行不同配置的综合测试
bash test_sdd_lskd.sh
```

### 完整训练套件

本项目提供了三个完整的训练脚本，涵盖不同的实验配置：

```bash
# 第一部分：ResNet32x4 -> ResNet8x4 基础实验和消融研究
bash start_sdd_lskd_training_part1.sh

# 第二部分：异构网络对实验
bash start_sdd_lskd_training_part2.sh  

# 第三部分：完整研究表格验证实验
bash start_sdd_lskd_training_part3.sh
```

## ⚙️ 配置说明

### 多尺度设置
- `M=[1]`：全局蒸馏 + LSKD 标准化
- `M=[1,2]`：全局 + 2×2 区域蒸馏 + LSKD
- `M=[1,2,4]`：全局 + 2×2 + 4×4 区域蒸馏 + LSKD（推荐）

### 配置文件示例

```yaml
DISTILLER:
  TYPE: "SDD_KD_LSKD"     # 主要方法
  TEACHER: "resnet32x4_sdd"
  STUDENT: "resnet8x4_sdd"
  
KD:
  TEMPERATURE: 2.0    # KD 温度参数
  LOSS:
    CE_WEIGHT: 1.0    # 交叉熵损失权重
    KD_WEIGHT: 9.0    # KD 损失权重

SOLVER:
  EPOCHS: 240
  BATCH_SIZE: 64
  LR: 0.05
```

## 📊 实验概述

### 支持的网络架构对
- **同构对**：ResNet32x4 → ResNet8x4
- **异构对**：
  - ResNet32x4 → ShuffleNetV1
  - ResNet32x4 → MobileNetV2
  - WideResNet-40-2 → VGG8
  - WideResNet-40-2 → ShuffleNetV1
  - WideResNet-40-2 → MobileNetV2

### 预期性能提升
基于 SDD-LSKD 融合方法：
- **M=[1]**：基线 LSKD 性能
- **M=[1,2]**：相比 M=[1] 提升 1-2%
- **M=[1,2,4]**：最佳性能，相比基线提升 2-3%

## 📁 项目结构

```
SDD-LSKD-Fusion/
├── mdistiller/                 # 核心蒸馏框架
│   └── distillers/
│       ├── SDD_KD_LSKD.py     # SDD+KD+LSKD 融合实现
│       └── SDD_DKD_LSKD.py    # SDD+DKD+LSKD 实现（暂不重点关注）
├── configs/                    # 配置文件
│   └── cifar100/
│       ├── sdd_kd_lskd/       # SDD+KD+LSKD 配置（主要）
│       └── sdd_dkd_lskd/      # SDD+DKD+LSKD 配置
├── start_sdd_lskd_training_part*.sh  # 训练脚本套件
├── test_sdd_lskd.sh           # 测试脚本
├── sdd-lskd-fusion.yml        # Conda 环境配置
├── requirements.txt            # Python 依赖
└── README_en.md               # 英文说明文档
```

## 🔬 技术细节

### 核心实现

融合方法在 `mdistiller/distillers/` 中实现，包含关键组件：

1. **多尺度特征提取**：继承 SDD 的空间金字塔池化
2. **Logit 标准化**：应用 LSKD 的 Z-score 标准化
3. **自适应损失权重**：结合一致性/互补性知识分类
4. **统一温度管理**：跨尺度的温度处理

### 关键函数
- `normalize_logit()`: Z-score logit 标准化
- `multi_scale_distillation_with_lskd()`: 多尺度蒸馏损失
- `kd_loss_with_lskd()`: 带 LSKD 的 KD 损失

## 🔬 方法变体

- **SDD_KD_LSKD**: SDD + KD + LSKD（主要关注）
- **SDD_DKD_LSKD**: SDD + DKD + LSKD（已实现，暂不重点关注）

## 📚 相关仓库

### 原始方法实现
- **LSKD (Logit Standardization)**: [logit-standardization-KD](https://github.com/sunshangquan/logit-standardization-KD)
- **SDD (Scale Decoupled Distillation)**: [SDD-CVPR2024](https://github.com/shicaiwei123/SDD-CVPR2024)
- **MDDistiller**: [Knowledge Distillation Framework](https://github.com/megvii-research/mdistiller)

### 本项目仓库
- **SDD-LSKD Fusion**: [SDD-LSKD-Fusion](https://github.com/Shardzzi/SDD-LSKD-Fusion)

## 📄 许可证

本项目使用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙋‍♂️ 支持

如有问题，请：
- 查看 [问题页面](https://github.com/Shardzzi/SDD-LSKD-Fusion/issues)
- 创建新问题描述您的情况
- 联系维护者

## 🎉 致谢

- 感谢 [SDD](https://github.com/shicaiwei123/SDD-CVPR2024) 和 [LSKD](https://github.com/sunshangquan/logit-standardization-KD) 作者的优秀研究
- 基于 [mdistiller 框架](https://github.com/megvii-research/mdistiller) 构建
- 感谢知识蒸馏研究社区的启发

## 📝 状态和说明

- **当前重点**: 专注于 SDD+KD+LSKD 方法
- **方法说明**: 使用传统 KD (Knowledge Distillation) 作为基础蒸馏方法
- **DKD 状态**: 虽然已实现 SDD+DKD+LSKD，但现阶段不作为重点探索
- **CUDA 要求**: 12.4（所有依赖库必须兼容）
- **环境配置**: conda → CUDA 12.4 → PyTorch → pip 依赖
- **未来工作**: 其他方法将在后续添加

---

**状态**: ✅ 已成功验证 CUDA 12.4 支持和有效的知识蒸馏流水线

**注意**: 本次实验专注于 SDD+KD+LSKD 方法，DKD 等其他方法待后续研究。

**English Documentation**: 详细的英文文档请参考 [README_en.md](README_en.md)
