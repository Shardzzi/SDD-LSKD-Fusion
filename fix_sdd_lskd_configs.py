#!/usr/bin/env python3
"""
Script to fix SDD-LSKD configuration files by replacing standard model names with SDD model names.
"""

import os
import glob
import re

def fix_config_file(filepath):
    """Fix a single configuration file by replacing model names with SDD versions."""
    print(f"Processing: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Model name mappings (standard -> SDD)
    model_mappings = {
        # ResNet models
        'resnet32x4': 'resnet32x4_sdd',
        'resnet8x4': 'resnet8x4_sdd',
        'resnet56': 'resnet56_sdd',
        'resnet20': 'resnet20_sdd',
        'resnet32': 'resnet32_sdd',
        'resnet110': 'resnet110_sdd',
        'resnet50': 'resnet50_sdd',
        
        # WideResNet models  
        'wrn_40_2': 'wrn_40_2_sdd',
        'wrn_16_2': 'wrn_16_2_sdd',
        'wrn_40_1': 'wrn_40_1_sdd',
        
        # VGG models
        'vgg13': 'vgg13_sdd',
        'vgg8': 'vgg8_sdd',
        
        # MobileNet and Shuffle models (these might not have SDD versions)
        # We'll handle them separately if needed
    }
    
    # Replace model names in TEACHER and STUDENT fields
    for old_name, new_name in model_mappings.items():
        # Match TEACHER: "model_name"
        pattern = f'TEACHER: "{old_name}"'
        replacement = f'TEACHER: "{new_name}"'
        content = content.replace(pattern, replacement)
        
        # Match STUDENT: "model_name"  
        pattern = f'STUDENT: "{old_name}"'
        replacement = f'STUDENT: "{new_name}"'
        content = content.replace(pattern, replacement)
    
    # Write back if changed
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  ✅ Updated: {os.path.basename(filepath)}")
        return True
    else:
        print(f"  ⚠️  No changes needed: {os.path.basename(filepath)}")
        return False

def main():
    """Main function to process all SDD-LSKD config files."""
    config_dir = "/root/repos/SDD-LSKD-Fusion/configs/cifar100/sdd_lskd/"
    
    # Find all YAML files in the SDD-LSKD directory
    yaml_files = glob.glob(os.path.join(config_dir, "*.yaml"))
    
    print(f"Found {len(yaml_files)} SDD-LSKD configuration files to process:")
    print("=" * 60)
    
    updated_files = 0
    for yaml_file in sorted(yaml_files):
        if fix_config_file(yaml_file):
            updated_files += 1
    
    print("=" * 60)
    print(f"✅ Processing complete! Updated {updated_files} out of {len(yaml_files)} files.")
    
    # Show summary of files that might need manual review
    print("\n📋 Files that might need manual review for non-standard models:")
    for yaml_file in sorted(yaml_files):
        basename = os.path.basename(yaml_file)
        if any(model in basename.lower() for model in ['mv2', 'mobile', 'shuffle', 'shu']):
            print(f"  - {basename}")

if __name__ == "__main__":
    main()
