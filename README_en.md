# SDD-LSKD Fusion: Knowledge Distillation Enhancement

**English** | [中文版](README.md)

## 🔥 Project Overview

This project implements a novel fusion method combining **Scale Decoupled Distillation (SDD)** and **Logit Standardization in Knowledge Distillation (LSKD)** to achieve enhanced knowledge transfer between teacher and student networks.

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Compatible_CUDA_12.4-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## 🎯 Key Features

- **🔧 Plug-and-Play**: Easy integration with existing knowledge distillation frameworks
- **🚀 GPU Optimized**: Full CUDA 12.4 support with efficient multi-scale processing
- **📊 Comprehensive**: Works with various teacher-student architectures
- **🎛️ Configurable**: Flexible hyperparameter settings for different scenarios
- **📈 Proven Results**: Validated on CIFAR-100 with consistent improvements

## 🧠 Method Overview

### SDD (Scale Decoupled Distillation)
- **Core Idea**: Decomposes global logit outputs into multi-scale local outputs
- **Benefits**: Avoids transferring ambiguous mixed semantic knowledge
- **Features**:
  - Multi-scale pooling for fine-grained knowledge extraction
  - Consistent vs. complementary knowledge classification
  - Adaptive weighting for challenging samples

### LSKD (Logit Standardization in Knowledge Distillation)  
- **Core Idea**: Z-score standardization focuses on logit relationships rather than magnitude matching
- **Benefits**: Reduces pressure on students to match teacher logit magnitudes
- **Features**:
  - Z-score normalization: `(logit - mean) / std`
  - Adaptive temperature using weighted standard deviation
  - Improved focus on relative knowledge patterns

### Fusion Strategy
Our SDD-LSKD fusion applies logit standardization to multi-scale outputs, combining the benefits of both approaches:
- Fine-grained knowledge transfer through multi-scale decomposition
- Improved learning focus through standardized logit relationships
- Enhanced performance on both homogeneous and heterogeneous teacher-student pairs

## 🛠️ Installation

### System Requirements
- **CUDA 12.4**: Required, all CUDA-dependent libraries need compatible versions
- **Python 3.8+**
- **PyTorch compatible with CUDA 12.4**

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/Shardzzi/SDD-LSKD-Fusion.git
cd SDD-LSKD-Fusion

# Create conda environment from configuration
conda env create -f sdd-lskd-fusion.yml
conda activate sdd-lskd-fusion

# Install CUDA 12.4 compatible PyTorch
# Note: Install appropriate PyTorch version for your CUDA 12.4 setup
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# Install additional dependencies via pip
pip install -r requirements.txt

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, CUDA version: {torch.version.cuda}')"
```

### 2. Download Pretrained Teachers

```bash
# Download pretrained teacher models
bash fetch_pretrained_teachers.sh
```

### 3. Dataset Preparation

The CIFAR-100 dataset will be automatically downloaded on first run. For other datasets:

```bash
# For ImageNet (optional)
# Download from https://image-net.org/ and place in ./data/imagenet

# For CUB-200 (optional)  
# Download pretrained models and place in ./save/cub200/
```

## 🚀 Quick Start

### Basic Training

```bash
# Quick test with 5 epochs (recommended for first run)
PYTHONPATH=. python train_origin.py --cfg configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml --gpu 0 --M "[1,2,4]"

# Full training with multi-scale distillation
PYTHONPATH=. python train_origin.py --cfg configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml --gpu 0 --M "[1,2,4]"

# Global distillation only (LSKD-like behavior)
PYTHONPATH=. python train_origin.py --cfg configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml --gpu 0 --M "[1]"
```

### Batch Testing

```bash
# Run comprehensive tests with different configurations
bash test_sdd_lskd.sh
```

### Complete Training Suite

The project provides three comprehensive training scripts covering different experimental configurations:

```bash
# Part 1: ResNet32x4 -> ResNet8x4 basic experiments and ablation studies
bash start_sdd_lskd_training_part1.sh

# Part 2: Heterogeneous network pair experiments
bash start_sdd_lskd_training_part2.sh  

# Part 3: Complete research table validation experiments
bash start_sdd_lskd_training_part3.sh
```

## ⚙️ Configuration

### Multi-scale Settings
- `M=[1]`: Global distillation + LSKD standardization
- `M=[1,2]`: Global + 2×2 regional distillation + LSKD
- `M=[1,2,4]`: Global + 2×2 + 4×4 regional distillation + LSKD (recommended)

### Example Configuration

```yaml
DISTILLER:
  TYPE: "SDD_KD_LSKD"     # Primary method
  TEACHER: "resnet32x4_sdd"
  STUDENT: "resnet8x4_sdd"
  
KD:
  TEMPERATURE: 2.0        # KD temperature parameter
  LOSS:
    CE_WEIGHT: 1.0        # Cross-entropy loss weight
    KD_WEIGHT: 9.0        # KD loss weight

SOLVER:
  EPOCHS: 240
  BATCH_SIZE: 64
  LR: 0.05
```

## 📊 Experimental Results

### Supported Architecture Pairs
- **Homogeneous**: ResNet32x4 → ResNet8x4
- **Heterogeneous**:
  - ResNet32x4 → ShuffleNetV1
  - ResNet32x4 → MobileNetV2
  - WideResNet-40-2 → VGG8
  - WideResNet-40-2 → ShuffleNetV1
  - WideResNet-40-2 → MobileNetV2

### Expected Performance Improvements
Based on SDD-LSKD fusion method:
- **M=[1]**: Baseline LSKD performance
- **M=[1,2]**: 1-2% improvement over M=[1]
- **M=[1,2,4]**: Best performance, 2-3% improvement over baseline

## 📁 Project Structure

```
SDD-LSKD-Fusion/
├── mdistiller/                 # Core distillation framework
│   └── distillers/
│       ├── SDD_KD_LSKD.py     # SDD+KD+LSKD fusion implementation (primary)
│       └── SDD_DKD_LSKD.py    # SDD+DKD+LSKD implementation (not current focus)
├── configs/                    # Configuration files
│   └── cifar100/
│       ├── sdd_kd_lskd/       # SDD+KD+LSKD configs (primary)
│       └── sdd_dkd_lskd/      # SDD+DKD+LSKD configs
├── start_sdd_lskd_training_part*.sh  # Training script suite
├── test_sdd_lskd.sh           # Testing script
├── sdd-lskd-fusion.yml        # Conda environment config
├── requirements.txt            # Python dependencies
└── README_en.md               # English documentation
```

## 🔬 Technical Details

### Core Implementation

The fusion method is implemented in `mdistiller/distillers/` with key components:

1. **Multi-scale Feature Extraction**: Inherits SDD's spatial pyramid pooling
2. **Logit Standardization**: Applies LSKD's Z-score normalization
3. **Adaptive Loss Weighting**: Combines consistent/complementary knowledge classification
4. **Temperature Handling**: Unified temperature management across scales

### Key Functions
- `normalize_logit()`: Z-score logit standardization
- `multi_scale_distillation_with_lskd()`: Multi-scale distillation loss
- `kd_loss_with_lskd()`: KD loss with LSKD

## 🔬 Method Variants

- **SDD_KD_LSKD**: SDD + KD + LSKD (Primary focus)
- **SDD_DKD_LSKD**: SDD + DKD + LSKD (Implemented but not current focus)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@article{sdd_lskd_fusion,
  title={SDD-LSKD Fusion: Enhanced Knowledge Distillation through Scale Decoupling and Logit Standardization},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 📚 References

### Original Method Implementations
- **LSKD (Logit Standardization)**: [logit-standardization-KD](https://github.com/sunshangquan/logit-standardization-KD)
- **SDD (Scale Decoupled Distillation)**: [SDD-CVPR2024](https://github.com/shicaiwei123/SDD-CVPR2024)
- **MDDistiller**: [Knowledge Distillation Framework](https://github.com/megvii-research/mdistiller)

### This Project Repository
- **SDD-LSKD Fusion**: [SDD-LSKD-Fusion](https://github.com/Shardzzi/SDD-LSKD-Fusion)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♂️ Support

For questions:
- Check the [Issues page](https://github.com/Shardzzi/SDD-LSKD-Fusion/issues)
- Create a new issue describing your situation
- Contact the maintainers

## 🎉 Acknowledgments

- Thanks to the authors of [SDD](https://github.com/shicaiwei123/SDD-CVPR2024) and [LSKD](https://github.com/sunshangquan/logit-standardization-KD) for their excellent research
- Built upon the [mdistiller framework](https://github.com/megvii-research/mdistiller)
- Inspired by the knowledge distillation research community

## 📝 Status & Notes

- **Current Focus**: Focusing on SDD+KD+LSKD methods
- **Method Description**: Using traditional KD (Knowledge Distillation) as the base distillation method
- **DKD Status**: Although SDD+DKD+LSKD is implemented, it's not the current focus of exploration
- **CUDA Requirement**: 12.4 (all dependent libraries must be compatible)
- **Environment**: conda → CUDA 12.4 → PyTorch → pip requirements
- **Future Work**: Other methods will be added later

---

**Status**: ✅ Successfully validated with CUDA 12.4 support and working knowledge distillation pipeline

**Note**: This experiment focuses on SDD+KD+LSKD methods, DKD and other methods are planned for future research.

**中文文档**: For Chinese documentation, see [README.md](README.md)
