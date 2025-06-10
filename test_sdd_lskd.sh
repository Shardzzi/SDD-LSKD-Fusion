#!/bin/bash

# SDD-LSKD融合方法测试脚本
# 测试不同的多尺度设置和logit标准化组合

echo "开始测试SDD-LSKD融合方法..."

# 测试1: 全局蒸馏 + LSKD标准化
echo "测试1: 全局蒸馏 + LSKD标准化 (M=[1])"
python train_origin.py --cfg configs/cifar100/sdd_lskd/res32x4_res8x4.yaml --gpu 0 --M [1] --warm_up 20

# 测试2: 多尺度蒸馏 + LSKD标准化  
echo "测试2: 多尺度蒸馏 + LSKD标准化 (M=[1,2])"
python train_origin.py --cfg configs/cifar100/sdd_lskd/res32x4_res8x4.yaml --gpu 0 --M [1,2] --warm_up 20

# 测试3: 更多尺度 + LSKD标准化
echo "测试3: 更多尺度 + LSKD标准化 (M=[1,2,4])"
python train_origin.py --cfg configs/cifar100/sdd_lskd/res32x4_res8x4.yaml --gpu 0 --M [1,2,4] --warm_up 20

# 测试不同的教师-学生对
echo "测试4: ResNet32x4 -> ShuffleNetV1 (M=[1,2,4])"
python train_origin.py --cfg configs/cifar100/sdd_lskd/res32x4_shuv1.yaml --gpu 0 --M [1,2,4] --warm_up 20

echo "测试5: ResNet32x4 -> MobileNetV2 (M=[1,2,4])"  
python train_origin.py --cfg configs/cifar100/sdd_lskd/res32x4_mv2.yaml --gpu 0 --M [1,2,4] --warm_up 20

echo "SDD-LSKD融合方法测试完成！"
