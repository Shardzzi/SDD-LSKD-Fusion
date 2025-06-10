# SDD-LSKD实验运行指南

本文档介绍如何使用提供的脚本来运行SDD-LSKD融合方法的实验。

## 脚本概览

### 1. `quick_validation.sh` - 快速验证脚本
**用途**: 快速验证SDD-LSKD方法是否正常工作
**运行时间**: 约15-30分钟
**包含测试**: 
- 全局LSKD蒸馏 (M=[1])
- 双尺度SDD-LSKD融合 (M=[1,2])  
- 三尺度SDD-LSKD融合 (M=[1,2,4])

```bash
./quick_validation.sh
```

### 2. `run_single_experiment.sh` - 单实验运行脚本
**用途**: 运行单个指定的实验
**灵活性**: 可自定义配置文件、多尺度设置、实验名称等

```bash
# 基本用法
./run_single_experiment.sh <config_file> <M_setting> [experiment_name] [gpu_id]

# 示例
./run_single_experiment.sh configs/cifar100/sdd_lskd/quick_test.yaml "[1,2,4]" "test_experiment" 0
./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml "[1]" "global_lskd"
```

### 3. `run_complete_experiments.sh` - 完整实验套件
**用途**: 运行完整的实验集合
**选项**: quick, full, ablation, comparison, all

```bash
# 显示帮助
./run_complete_experiments.sh --help

# 运行快速测试
./run_complete_experiments.sh --set quick

# 运行完整实验（240轮训练）
./run_complete_experiments.sh --set full --gpu 0

# 运行消融研究
./run_complete_experiments.sh --set ablation

# 运行所有实验类型
./run_complete_experiments.sh --set all
```

## 实验配置文件

### CIFAR-100数据集配置
- `configs/cifar100/sdd_lskd/quick_test.yaml` - 快速测试(5轮)
- `configs/cifar100/sdd_lskd/res32x4_res8x4.yaml` - ResNet32x4→ResNet8x4(240轮)
- `configs/cifar100/sdd_lskd/res32x4_mv2.yaml` - ResNet32x4→MobileNetV2(240轮)
- `configs/cifar100/sdd_lskd/res32x4_shuv1.yaml` - ResNet32x4→ShuffleNetV1(240轮)

## 多尺度设置说明

### M参数解释
- `[1]` - 仅全局蒸馏，使用LSKD标准化
- `[1,2]` - 双尺度：全局+2x2块，SDD+LSKD融合
- `[1,2,4]` - 三尺度：全局+2x2+4x4块，完整SDD+LSKD融合
- `[2]` - 仅2x2块蒸馏（消融研究）
- `[4]` - 仅4x4块蒸馏（消融研究）

## 推荐使用流程

### 1. 环境验证
首先运行快速验证确保环境正常：
```bash
./quick_validation.sh
```

### 2. 单个实验测试
测试特定配置：
```bash
./run_single_experiment.sh configs/cifar100/sdd_lskd/quick_test.yaml "[1,2,4]" "fusion_test"
```

### 3. 完整实验
运行完整的实验集合：
```bash
# 先运行快速测试确认
./run_complete_experiments.sh --set quick

# 然后运行完整实验
./run_complete_experiments.sh --set full
```

## 日志和结果

### 日志目录结构
```
logs/
├── quick_validation/          # 快速验证日志
├── single_experiments/        # 单实验日志
├── sdd_lskd_quick/           # 快速测试集日志
├── sdd_lskd_full/            # 完整实验日志
├── sdd_lskd_ablation/        # 消融研究日志
├── sdd_lskd_comparison/      # 基线对比日志
└── summaries/                # 实验总结
```

### 结果查看
每个实验的最终准确率会记录在对应的日志文件中。完整实验套件还会生成总结报告。

## 故障排除

### 常见问题
1. **conda环境激活失败**: 确保conda已安装且环境名称正确
2. **CUDA不可用**: 检查GPU驱动和CUDA安装
3. **模块导入错误**: 确保所有依赖包已安装
4. **超时错误**: 可调整脚本中的超时设置

### 调试技巧
- 查看详细日志文件了解错误信息
- 使用quick_test.yaml进行快速调试
- 检查GPU内存使用情况
- 确认数据集已正确下载

## 实验预期结果

基于SDD-LSKD融合方法，预期结果：
- M=[1]: 基线LSKD性能
- M=[1,2]: 相比M=[1]有1-2%提升
- M=[1,2,4]: 最佳性能，相比基线有2-3%提升

具体数值因架构组合而异，详细结果请参考生成的实验报告。
