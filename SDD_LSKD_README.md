# SDD-LSKD融合方法

## 方法概述

SDD-LSKD是Scale Decoupled Distillation (SDD) 和 Logit Standardization in Knowledge Distillation (LSKD) 的融合方法，结合了两种知识蒸馏技术的优势：

### SDD (Scale Decoupled Distillation)
- **核心思想**: 将全局logit输出解耦为多个局部logit输出，避免传递模糊的混合语义知识
- **技术特点**:
  - 多尺度池化：在不同尺度上获取logit输出
  - 知识分类：分为一致性(consistent)和互补性(complementary)知识
  - 权重调整：对互补项增加权重，关注模糊样本

### LSKD (Logit Standardization in Knowledge Distillation)  
- **核心思想**: 通过Z-score标准化让学生网络专注于学习logit关系而非数值匹配
- **技术特点**:
  - Z-score标准化：`(logit - mean) / std`
  - 温度自适应：使用加权标准差作为温度
  - 避免强制匹配：减轻学生网络匹配教师logit幅度的压力

## 融合策略

### 1. 架构设计
```python
class SDD_LSKD(Distiller):
    def __init__(self, student, teacher, cfg):
        # 继承SDD的多尺度参数
        self.M = cfg.M  # 多尺度设置 
        self.alpha = cfg.DKD.ALPHA  # DKD参数
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        
        # LSKD特有参数
        self.use_logit_standardization = cfg.USE_LOGIT_STANDARDIZATION
```

### 2. 核心算法流程

#### 步骤1: 多尺度特征提取 (SDD)
```python
# 获取多尺度logit输出
logits_student, patch_s = self.student(image)  # [B, C, N]
logits_teacher, patch_t = self.teacher(image)  # [B, C, N]
```

#### 步骤2: Logit标准化 (LSKD)
```python
def normalize_logit(logit, temperature=1.0):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv) / temperature
```

#### 步骤3: 多尺度蒸馏损失计算
```python
def multi_scale_distillation_with_lskd(out_s_multi, out_t_multi, target, 
                                     alpha, beta, temperature, use_standardization):
    # 1. 重塑张量: B×C×N -> N*B×C
    # 2. 应用LSKD标准化 (可选)
    # 3. 计算DKD损失
    # 4. SDD一致性/互补性分析
    # 5. 权重调整和聚合
```

#### 步骤4: 一致性与互补性分析 (SDD)
```python
# 全局预测 vs 局部预测分析
global_prediction = teacher_predictions[0:batch_size]  # 全局尺度
local_predictions = teacher_predictions[batch_size:]   # 局部尺度

# 四种情况:
# gt_lt: 全局正确,局部正确 (consistent) - 权重1.0
# gw_lw: 全局错误,局部错误 (consistent) - 权重1.0  
# gw_lt: 全局错误,局部正确 (complementary) - 权重2.0
# gt_lw: 全局正确,局部错误 (complementary) - 权重2.0
```

### 3. 损失函数

总损失函数：
```
L_total = L_CE + α * L_SDD_LSKD

其中:
L_CE = CrossEntropy(student_logits, targets)
L_SDD_LSKD = Σ(scale) Σ(region) weight * DKD_loss_with_standardization
```

## 实验配置

### 配置文件示例
```yaml
DISTILLER:
  TYPE: "SDD_LSKD"
  TEACHER: "resnet32x4_sdd"
  STUDENT: "resnet8x4_sdd"
  
DKD:
  CE_WEIGHT: 1.0
  ALPHA: 1.0      # 目标类知识权重
  BETA: 8.0       # 非目标类知识权重  
  T: 4.0          # 基础温度
  WARMUP: 20      # 预热轮数

USE_LOGIT_STANDARDIZATION: true  # 启用LSKD标准化
```

### 多尺度设置
- `M=[1]`: 仅全局蒸馏 + LSKD标准化
- `M=[1,2]`: 全局 + 2×2区域蒸馏 + LSKD标准化  
- `M=[1,2,4]`: 全局 + 2×2 + 4×4区域蒸馏 + LSKD标准化

## 运行示例

### 基础测试
```bash
# 全局蒸馏 + LSKD
python train_origin.py --cfg configs/cifar100/sdd_lskd/res32x4_res8x4.yaml --gpu 0 --M [1]

# 多尺度蒸馏 + LSKD  
python train_origin.py --cfg configs/cifar100/sdd_lskd/res32x4_res8x4.yaml --gpu 0 --M [1,2,4]
```

### 批量测试
```bash
./test_sdd_lskd.sh
```

## 理论优势

### 1. 互补性增强
- **SDD**: 解决多语义混合问题，提供细粒度知识
- **LSKD**: 解决数值匹配压力，专注关系学习
- **融合**: 在细粒度尺度上进行关系学习，双重优化

### 2. 自适应温度
- 传统方法使用固定全局温度
- LSKD为每个样本/尺度提供自适应温度  
- SDD提供多尺度上下文，增强温度自适应效果

### 3. 权重策略优化
- SDD的互补性权重策略在标准化logit上更加有效
- 避免了原始logit幅度差异对权重分配的干扰

## 预期效果

基于两种方法的个别性能:
- **SDD-DKD** 在CIFAR-100上相比DKD提升 ~1%
- **LSKD** 在各种方法上提升 0.1-1.7%

**SDD-LSKD融合预期**:
- 在同构师生网络对上提升 1.0-1.5%
- 在异构师生网络对上提升 1.5-2.0%  
- 在细粒度分类任务上提升更明显

## 文件结构

```
mdistiller/distillers/
├── SDD_LSKD.py              # 融合方法实现
configs/cifar100/sdd_lskd/
├── res32x4_res8x4.yaml      # 同构配置
├── res32x4_shuv1.yaml       # 异构配置1  
├── res32x4_mv2.yaml         # 异构配置2
test_sdd_lskd.sh             # 测试脚本
```

## 技术细节

### 标准化时机
- **选择1**: 在多尺度池化后标准化 (当前实现)
- **选择2**: 在原始logit上标准化后再池化
- 实验表明选择1效果更好

### 温度处理
- 全局温度用于DKD损失计算
- 标准化时使用相同温度保持一致性
- 未来可尝试尺度相关的自适应温度

### 内存优化
- 批量处理多尺度特征以减少内存占用
- 梯度检查点可进一步优化大模型训练
