#!/usr/bin/env python3
"""
Update training scripts to use correct SDD_DKD_LSKD config paths
"""

import re
import glob

def update_training_script(script_path):
    """Update a single training script"""
    print(f"Updating {script_path}")
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Replace sdd_lskd with sdd_dkd_lskd
    updated_content = content.replace('configs/cifar100/sdd_lskd/', 'configs/cifar100/sdd_dkd_lskd/')
    
    # Also update comments and descriptions to be more accurate
    updated_content = updated_content.replace('SDD-LSKD', 'SDD-DKD-LSKD')
    updated_content = updated_content.replace('sdd_lskd_full', 'sdd_dkd_lskd_full')
    
    with open(script_path, 'w') as f:
        f.write(updated_content)
    
    print(f"  ✅ Updated {script_path}")

def main():
    print("Updating SDD-DKD-LSKD Training Scripts")
    print("=" * 50)
    
    # Find all training scripts
    training_scripts = glob.glob("/root/repos/SDD-LSKD-Fusion/start_sdd_lskd_training_part*.sh")
    
    for script in sorted(training_scripts):
        try:
            update_training_script(script)
        except Exception as e:
            print(f"Error updating {script}: {e}")
    
    print("\n✅ All training scripts updated!")
    print("Scripts now use configs/cifar100/sdd_dkd_lskd/ directory")

if __name__ == "__main__":
    main()
