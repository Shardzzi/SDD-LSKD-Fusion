#!/usr/bin/env python3
"""
Update all SDD+KD+LSKD config files to use LSKD standard parameters:

For DKD-based configs:
- T: 2.0 (base-temp)
- ALPHA: 9.0 (1.0 * 9, kd-weight scaling)  
- BETA: 72.0 (8.0 * 9, kd-weight scaling)

For KD-based configs:
- TEMPERATURE: 2.0 (base-temp)
- KD_WEIGHT: 9.0 (kd-weight scaling)

This ensures consistency with logit-standardization-KD paper settings.
"""

import os
import yaml
import glob

def update_config_file(config_path):
    """Update a single config file with LSKD standard parameters"""
    print(f"Updating {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update DKD parameters to match LSKD standard (for SDD_DKD_LSKD)
    if 'DKD' in config:
        config['DKD']['T'] = 2.0      # base-temp
        config['DKD']['ALPHA'] = 9.0  # 1.0 * 9 (kd-weight)
        config['DKD']['BETA'] = 72.0  # 8.0 * 9 (kd-weight)
        
        print(f"  Updated DKD params: T={config['DKD']['T']}, ALPHA={config['DKD']['ALPHA']}, BETA={config['DKD']['BETA']}")
    
    # Update KD parameters to match LSKD standard (for SDD_KD_LSKD)
    if 'KD' in config:
        config['KD']['TEMPERATURE'] = 2.0    # base-temp
        config['KD']['LOSS']['KD_WEIGHT'] = 9.0  # kd-weight
        
        print(f"  Updated KD params: TEMPERATURE={config['KD']['TEMPERATURE']}, KD_WEIGHT={config['KD']['LOSS']['KD_WEIGHT']}")
    
    # Write back to file
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def main():
    print("Updating SDD+KD+LSKD configs to LSKD standard parameters")
    print("=" * 70)
    
    # Find all SDD+LSKD config files in both directories
    sdd_dkd_lskd_configs = glob.glob("/root/repos/SDD-LSKD-Fusion/configs/cifar100/sdd_dkd_lskd/*.yaml")
    sdd_kd_lskd_configs = glob.glob("/root/repos/SDD-LSKD-Fusion/configs/cifar100/sdd_kd_lskd/*.yaml")
    
    all_configs = sdd_dkd_lskd_configs + sdd_kd_lskd_configs
    
    print(f"Found {len(sdd_dkd_lskd_configs)} SDD_DKD_LSKD config files")
    print(f"Found {len(sdd_kd_lskd_configs)} SDD_KD_LSKD config files") 
    print(f"Total: {len(all_configs)} config files to update")
    
    for config_path in sorted(all_configs):
        try:
            update_config_file(config_path)
        except Exception as e:
            print(f"Error updating {config_path}: {e}")
    
    print("\n" + "=" * 70) 
    print("✅ All SDD+KD+LSKD configs updated to LSKD standard parameters!")
    print("Parameters now match logit-standardization-KD:")
    print("  DKD-based: T=2.0, ALPHA=9.0, BETA=72.0")
    print("  KD-based: TEMPERATURE=2.0, KD_WEIGHT=9.0")

if __name__ == "__main__":
    main()
