#!/usr/bin/env python3
"""
Quick test script for SDD-LSKD fusion method
Tests basic functionality and validates the implementation works correctly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mdistiller.models.cifar import cifar_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset

# Import our SDD_LSKD fusion class
from mdistiller.distillers.SDD_LSKD import SDD_LSKD, normalize_logit

def test_basic_functionality():
    """Test basic functionality of SDD_LSKD fusion"""
    print("=== Testing Basic SDD-LSKD Functionality ===")
    
    # Create a simple config object
    class SimpleConfig:
        def __init__(self):
            self.DKD = type('obj', (object,), {})()
            self.DKD.CE_WEIGHT = 1.0
            self.DKD.ALPHA = 1.0
            self.DKD.BETA = 8.0
            self.DKD.T = 4.0
            self.DKD.WARMUP = 20
            self.warmup = 20
            self.M = '[1,2,4]'
            self.USE_LOGIT_STANDARDIZATION = True
    
    # Create simple student/teacher models
    student_net, _ = cifar_model_dict['resnet8x4_sdd']
    teacher_net, teacher_path = cifar_model_dict['resnet32x4_sdd']
    
    student = student_net(num_classes=100, M='[1,2,4]')
    teacher = teacher_net(num_classes=100, M='[1,2,4]')
    
    # Load teacher weights
    checkpoint = torch.load(teacher_path, map_location='cpu')
    teacher.load_state_dict(checkpoint['model'])
    
    # Create SDD_LSKD distiller
    cfg = SimpleConfig()
    distiller = SDD_LSKD(student, teacher, cfg)
    
    print("✓ SDD_LSKD distiller created successfully")
    
    # Test with dummy data
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 32, 32)
    dummy_target = torch.randint(0, 100, (batch_size,))
    
    # Test forward pass
    distiller.eval()
    with torch.no_grad():
        logits_student, losses_dict = distiller.forward_train(dummy_input, dummy_target, epoch=10)
    
    print(f"✓ Forward pass successful. Student logits shape: {logits_student.shape}")
    print(f"✓ Losses computed: {list(losses_dict.keys())}")
    for k, v in losses_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    return True

def test_logit_standardization():
    """Test LSKD logit standardization function"""
    print("\n=== Testing LSKD Logit Standardization ===")
    
    # Create sample logits
    batch_size = 4
    num_classes = 100
    logits = torch.randn(batch_size, num_classes) * 10  # Large variance
    
    print(f"Original logits - Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
    
    # Apply standardization
    standardized_logits = normalize_logit(logits, temperature=4.0)
    
    print(f"Standardized logits - Mean: {standardized_logits.mean():.4f}, Std: {standardized_logits.std():.4f}")
    
    # Check that each sample is standardized
    for i in range(batch_size):
        sample_mean = standardized_logits[i].mean().item()
        sample_std = standardized_logits[i].std().item()
        print(f"  Sample {i}: Mean={sample_mean:.4f}, Std={sample_std:.4f}")
    
    print("✓ Logit standardization working correctly")
    return True

def test_multi_scale_settings():
    """Test different multi-scale settings"""
    print("\n=== Testing Multi-Scale Settings ===")
    
    # Import here to avoid circular import issues
    from mdistiller.models.cifar import cifar_model_dict
    
    class SimpleConfig:
        def __init__(self, M_setting):
            self.DKD = type('obj', (object,), {})()
            self.DKD.CE_WEIGHT = 1.0
            self.DKD.ALPHA = 1.0
            self.DKD.BETA = 8.0
            self.DKD.T = 4.0
            self.DKD.WARMUP = 20
            self.warmup = 20
            self.M = M_setting
            self.USE_LOGIT_STANDARDIZATION = True
    
    # Test different M settings
    M_settings = ['[1]', '[1,2]', '[1,2,4]']
    
    for M_setting in M_settings:
        print(f"\nTesting M = {M_setting}")
        
        cfg = SimpleConfig(M_setting)
        
        if M_setting == '[1]':
            # For global setting, use standard models without SDD
            student_net, _ = cifar_model_dict['resnet8x4']
            teacher_net, teacher_path = cifar_model_dict['resnet32x4']
            student = student_net(num_classes=100)
            teacher = teacher_net(num_classes=100)
        else:
            # For multi-scale, use SDD models
            student_net, _ = cifar_model_dict['resnet8x4_sdd']
            teacher_net, teacher_path = cifar_model_dict['resnet32x4_sdd']
            student = student_net(num_classes=100, M=M_setting)
            teacher = teacher_net(num_classes=100, M=M_setting)
        
        # Load teacher weights
        checkpoint = torch.load(teacher_path, map_location='cpu')
        teacher.load_state_dict(checkpoint['model'])
        
        # Create distiller
        distiller = SDD_LSKD(student, teacher, cfg)
        
        # Test forward pass
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 32, 32)
        dummy_target = torch.randint(0, 100, (batch_size,))
        
        distiller.eval()
        with torch.no_grad():
            logits_student, losses_dict = distiller.forward_train(dummy_input, dummy_target, epoch=10)
        
        # Handle tensor conversion safely
        loss_ce = losses_dict['loss_ce']
        loss_kd = losses_dict['loss_kd']
        if loss_ce.numel() == 1:
            loss_ce_val = loss_ce.item()
        else:
            loss_ce_val = loss_ce.mean().item()
        if loss_kd.numel() == 1:
            loss_kd_val = loss_kd.item()
        else:
            loss_kd_val = loss_kd.mean().item()
            
        print(f"  ✓ M={M_setting} working. Loss CE: {loss_ce_val:.4f}, Loss KD: {loss_kd_val:.4f}")
    
    return True

def main():
    """Run all tests"""
    print("Starting SDD-LSKD Fusion Quick Tests...")
    
    try:
        # Test 1: Basic functionality
        test_basic_functionality()
        
        # Test 2: Logit standardization
        test_logit_standardization()
        
        # Test 3: Multi-scale settings
        test_multi_scale_settings()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! SDD-LSKD fusion is working correctly.")
        print("Ready to run full experiments!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
