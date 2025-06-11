#!/usr/bin/env python3
"""
Fix all SDD-DKD-LSKD config file TAGs to use correct naming
"""

import os
import yaml
import glob

def fix_config_tag(config_path):
    """Fix TAG in config file"""
    print(f"Fixing TAG in {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Fix TAG format for SDD_DKD_LSKD
    if 'sdd_dkd_lskd' in config_path:
        tag_parts = config['EXPERIMENT']['TAG'].split(',')
        if len(tag_parts) >= 3:
            config['EXPERIMENT']['TAG'] = f"sdd_dkd_lskd,{tag_parts[1]},{tag_parts[2]}"
        print(f"  Updated TAG to: {config['EXPERIMENT']['TAG']}")
    
    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def main():
    print("Fixing SDD-DKD-LSKD config file TAGs")
    print("=" * 50)
    
    # Find all SDD-DKD-LSKD config files
    sdd_dkd_lskd_configs = glob.glob("/root/repos/SDD-LSKD-Fusion/configs/cifar100/sdd_dkd_lskd/*.yaml")
    
    for config_path in sorted(sdd_dkd_lskd_configs):
        try:
            fix_config_tag(config_path)
        except Exception as e:
            print(f"Error fixing {config_path}: {e}")
    
    print("\n✅ All config TAGs fixed!")

if __name__ == "__main__":
    main()
