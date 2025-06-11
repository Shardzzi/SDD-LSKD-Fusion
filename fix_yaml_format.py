#!/usr/bin/env python3
"""
修复YAML配置文件格式问题
- 恢复字符串引号
- 修复列表格式
- 确保TAG正确
"""

import os
import yaml
import glob

def fix_config_file(config_path):
    """修复单个配置文件的格式问题"""
    print(f"修复 {config_path}")
    
    # 直接写入正确的内容，而不是依赖yaml.dump
    with open(config_path, 'r') as f:
        content = f.read()
    
    # 提取文件名来确定正确的配置
    filename = os.path.basename(config_path)
    config_dir = os.path.basename(os.path.dirname(config_path))
    
    if config_dir == 'sdd_dkd_lskd':
        fix_sdd_dkd_lskd_config(config_path, filename)
    elif config_dir == 'sdd_kd_lskd':
        fix_sdd_kd_lskd_config(config_path, filename)

def fix_sdd_dkd_lskd_config(config_path, filename):
    """修复SDD_DKD_LSKD配置文件"""
    
    # 根据文件名确定teacher和student
    config_map = {
        'res32x4_res8x4.yaml': {
            'tag': 'sdd_dkd_lskd,res32x4,res8x4',
            'teacher': 'resnet32x4_sdd',
            'student': 'resnet8x4_sdd'
        },
        'res32x4_shuv1.yaml': {
            'tag': 'sdd_dkd_lskd,res32x4,shuv1',
            'teacher': 'resnet32x4_sdd',
            'student': 'ShuffleV1_sdd'
        },
        'wrn40_2_shuv1.yaml': {
            'tag': 'sdd_dkd_lskd,wrn40_2,shuv1',
            'teacher': 'wrn_40_2_sdd',
            'student': 'ShuffleV1_sdd'
        },
        'res32x4_mv2.yaml': {
            'tag': 'sdd_dkd_lskd,res32x4,mv2',
            'teacher': 'resnet32x4_sdd',
            'student': 'MobileNetV2_sdd'
        },
        'wrn40_2_vgg8.yaml': {
            'tag': 'sdd_dkd_lskd,wrn40_2,vgg8',
            'teacher': 'wrn_40_2_sdd',
            'student': 'vgg8_sdd'
        },
        'wrn40_2_mv2.yaml': {
            'tag': 'sdd_dkd_lskd,wrn40_2,mv2',
            'teacher': 'wrn_40_2_sdd',
            'student': 'MobileNetV2_sdd'
        }
    }
    
    if filename not in config_map:
        print(f"未知的配置文件: {filename}")
        return
    
    info = config_map[filename]
    
    content = f'''EXPERIMENT:
  NAME: ""
  TAG: "{info['tag']}"
  PROJECT: "cifar100_baselines"
DISTILLER:
  TYPE: "SDD_DKD_LSKD"
  TEACHER: "{info['teacher']}"
  STUDENT: "{info['student']}"
SOLVER:
  BATCH_SIZE: 64
  EPOCHS: 240
  LR: 0.05
  LR_DECAY_STAGES: [150, 180, 210]
  LR_DECAY_RATE: 0.1
  WEIGHT_DECAY: 0.0005
  MOMENTUM: 0.9
  TYPE: "SGD"
DKD:
  CE_WEIGHT: 1.0
  ALPHA: 9.0
  BETA: 72.0
  T: 2.0
  WARMUP: 20
DATASET:
  TYPE: "cifar100"
  NUM_WORKERS: 2
'''
    
    with open(config_path, 'w') as f:
        f.write(content)
    print(f"  ✅ 修复完成: {filename}")

def fix_sdd_kd_lskd_config(config_path, filename):
    """修复SDD_KD_LSKD配置文件"""
    
    # 根据文件名确定teacher和student
    config_map = {
        'res32x4_res8x4.yaml': {
            'tag': 'sdd_kd_lskd,res32x4,res8x4',
            'teacher': 'resnet32x4_sdd',
            'student': 'resnet8x4_sdd'
        },
        'res32x4_shuv1.yaml': {
            'tag': 'sdd_kd_lskd,res32x4,shuv1',
            'teacher': 'resnet32x4_sdd',
            'student': 'ShuffleV1_sdd'
        },
        'wrn40_2_shuv1.yaml': {
            'tag': 'sdd_kd_lskd,wrn40_2,shuv1',
            'teacher': 'wrn_40_2_sdd',
            'student': 'ShuffleV1_sdd'
        },
        'res32x4_mv2.yaml': {
            'tag': 'sdd_kd_lskd,res32x4,mv2',
            'teacher': 'resnet32x4_sdd',
            'student': 'MobileNetV2_sdd'
        },
        'wrn40_2_vgg8.yaml': {
            'tag': 'sdd_kd_lskd,wrn40_2,vgg8',
            'teacher': 'wrn_40_2_sdd',
            'student': 'vgg8_sdd'
        },
        'wrn40_2_mv2.yaml': {
            'tag': 'sdd_kd_lskd,wrn40_2,mv2',
            'teacher': 'wrn_40_2_sdd',
            'student': 'MobileNetV2_sdd'
        },
        'vgg13_vgg8.yaml': {
            'tag': 'sdd_kd_lskd,vgg13,vgg8',
            'teacher': 'vgg13_sdd',
            'student': 'vgg8_sdd'
        }
    }
    
    if filename not in config_map:
        print(f"未知的配置文件: {filename}")
        return
    
    info = config_map[filename]
    
    content = f'''EXPERIMENT:
  NAME: ""
  TAG: "{info['tag']}"
  PROJECT: "cifar100_baselines"
DISTILLER:
  TYPE: "SDD_KD_LSKD"
  TEACHER: "{info['teacher']}"
  STUDENT: "{info['student']}"
SOLVER:
  BATCH_SIZE: 64
  EPOCHS: 240
  LR: 0.05
  LR_DECAY_STAGES: [150, 180, 210]
  LR_DECAY_RATE: 0.1
  WEIGHT_DECAY: 0.0005
  MOMENTUM: 0.9
  TYPE: "SGD"
KD:
  TEMPERATURE: 2.0
  LOSS:
    CE_WEIGHT: 1.0
    KD_WEIGHT: 9.0
DATASET:
  TYPE: "cifar100"
  NUM_WORKERS: 2
'''
    
    with open(config_path, 'w') as f:
        f.write(content)
    print(f"  ✅ 修复完成: {filename}")

def main():
    print("修复所有SDD+LSKD配置文件格式")
    print("=" * 50)
    
    # 查找所有配置文件
    sdd_dkd_lskd_configs = glob.glob("/root/repos/SDD-LSKD-Fusion/configs/cifar100/sdd_dkd_lskd/*.yaml")
    sdd_kd_lskd_configs = glob.glob("/root/repos/SDD-LSKD-Fusion/configs/cifar100/sdd_kd_lskd/*.yaml")
    
    all_configs = sdd_dkd_lskd_configs + sdd_kd_lskd_configs
    
    print(f"找到 {len(sdd_dkd_lskd_configs)} 个SDD_DKD_LSKD配置文件")
    print(f"找到 {len(sdd_kd_lskd_configs)} 个SDD_KD_LSKD配置文件")
    print(f"总计: {len(all_configs)} 个配置文件")
    print()
    
    for config_path in sorted(all_configs):
        try:
            fix_config_file(config_path)
        except Exception as e:
            print(f"修复失败 {config_path}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ 所有配置文件格式修复完成！")
    print("修复内容:")
    print("  - 恢复字符串引号")
    print("  - 修复列表格式为 [150, 180, 210]")
    print("  - 确保TAG格式正确")
    print("  - 设置LSKD标准参数")

if __name__ == "__main__":
    main()
