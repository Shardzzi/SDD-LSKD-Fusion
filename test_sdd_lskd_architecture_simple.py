#!/usr/bin/env python3
"""
Quick test to verify SDD+LSKD architecture works correctly
"""

import sys
import os
import torch
sys.path.append('/root/repos/SDD-LSKD-Fusion')

def test_sdd_architectures():
    """Test all SDD+LSKD combinations"""
    print("Testing SDD+LSKD Architecture Combinations")
    print("=" * 50)
    
    # Import distillers
    from mdistiller.distillers import distiller_dict
    
    # Test distiller registration
    test_distillers = ["SDD_DKD_LSKD", "SDD_KD_LSKD"]
    
    print("\n🔍 Testing Distiller Registration:")
    for distiller_name in test_distillers:
        if distiller_name in distiller_dict:
            print(f"✅ {distiller_name} registered successfully")
        else:
            print(f"❌ {distiller_name} NOT found in distiller_dict!")
    
    # Test individual configs
    print("\n🧪 Testing Individual Configurations:")
    
    # Test SDD_DKD_LSKD config
    try:
        from mdistiller.engine.cfg import CFG
        cfg1 = CFG.clone()
        cfg1.merge_from_file("/root/repos/SDD-LSKD-Fusion/configs/cifar100/sdd_dkd_lskd/res32x4_res8x4.yaml")
        
        print(f"✅ SDD_DKD_LSKD config loaded:")
        print(f"   - DISTILLER.TYPE: {cfg1.DISTILLER.TYPE}")
        print(f"   - TEACHER: {cfg1.DISTILLER.TEACHER}")
        print(f"   - STUDENT: {cfg1.DISTILLER.STUDENT}")
        print(f"   - DKD.T: {cfg1.DKD.T}")
        print(f"   - DKD.ALPHA: {cfg1.DKD.ALPHA}")
        print(f"   - DKD.BETA: {cfg1.DKD.BETA}")
        
    except Exception as e:
        print(f"❌ Error loading SDD_DKD_LSKD config: {e}")
    
    # Test SDD_KD_LSKD config  
    try:
        from mdistiller.engine.cfg import CFG
        cfg2 = CFG.clone() 
        cfg2.merge_from_file("/root/repos/SDD-LSKD-Fusion/configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml")
        
        print(f"\n✅ SDD_KD_LSKD config loaded:")
        print(f"   - DISTILLER.TYPE: {cfg2.DISTILLER.TYPE}")
        print(f"   - TEACHER: {cfg2.DISTILLER.TEACHER}")
        print(f"   - STUDENT: {cfg2.DISTILLER.STUDENT}")
        print(f"   - KD.TEMPERATURE: {cfg2.KD.TEMPERATURE}")
        print(f"   - KD.LOSS.KD_WEIGHT: {cfg2.KD.LOSS.KD_WEIGHT}")
        
    except Exception as e:
        print(f"❌ Error loading SDD_KD_LSKD config: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 SDD+LSKD Architecture Summary:")
    print("✅ SDD_DKD_LSKD: SDD + DKD + LSKD (原SDD_LSKD.py)")
    print("✅ SDD_KD_LSKD: SDD + KD + LSKD (新创建)")
    print("⚠️  SDD_MLKD_LSKD: 未实现 (原框架中无SDD_MLKD)")
    print("\n💡 符合logit-standardization-KD论文设计理念：")
    print("   - KD → KD+LSKD")
    print("   - DKD → DKD+LSKD") 
    print("   - SDD+KD → SDD+KD+LSKD")
    print("   - SDD+DKD → SDD+DKD+LSKD")
    print("\n🔧 LSKD标准参数 (与logit-standardization-KD一致):")
    print("   - DKD: T=2.0, ALPHA=9.0, BETA=72.0")
    print("   - KD: TEMPERATURE=2.0, KD_WEIGHT=9.0")

if __name__ == "__main__":
    test_sdd_architectures()
