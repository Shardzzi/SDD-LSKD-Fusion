#!/usr/bin/env python3
"""
Test to verify our LSKD implementation matches the reference implementation
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Add paths for both implementations
sys.path.append('/root/repos/SDD-LSKD-Fusion')
sys.path.append('/root/repos/SDD-LSKD-Fusion/logit-standardization-KD')

# Import reference implementation
from logit_standardization_KD.mdistiller.distillers.DKD import normalize, dkd_loss, _get_gt_mask, _get_other_mask, cat_mask

# Import our implementation  
from mdistiller.distillers.SDD_LSKD import normalize_logit, dkd_loss_with_lskd

def test_normalization_consistency():
    """Test that both normalization functions produce the same results"""
    print("=== Testing Normalization Consistency ===")
    
    # Create test data
    torch.manual_seed(42)
    batch_size, num_classes = 16, 100
    logits = torch.randn(batch_size, num_classes)
    
    # Reference normalization
    ref_normalized = normalize(logits)
    
    # Our normalization
    our_normalized = normalize_logit(logits)
    
    # Check if they're identical
    max_diff = torch.max(torch.abs(ref_normalized - our_normalized)).item()
    print(f"Max difference in normalization: {max_diff}")
    
    if max_diff < 1e-6:
        print("✅ Normalization functions are consistent!")
    else:
        print("❌ Normalization functions differ!")
        
    return max_diff < 1e-6

def test_dkd_loss_consistency():
    """Test that both DKD loss functions produce the same results"""
    print("\n=== Testing DKD Loss Consistency ===")
    
    # Create test data
    torch.manual_seed(42)
    batch_size, num_classes = 16, 100
    logits_student = torch.randn(batch_size, num_classes)
    logits_teacher = torch.randn(batch_size, num_classes)
    target = torch.randint(0, num_classes, (batch_size,))
    
    # Parameters
    alpha, beta = 1.0, 8.0
    temperature = 4.0
    
    # Reference implementation (with logit standardization enabled)
    ref_loss = dkd_loss(
        logits_student, logits_teacher, target,
        alpha, beta, temperature, logit_stand=True
    )
    
    # Our implementation
    our_loss = dkd_loss_with_lskd(
        logits_student, logits_teacher, target,
        alpha, beta, temperature
    )
    
    # Check if they're identical
    diff = torch.abs(ref_loss - our_loss).item()
    print(f"Reference loss: {ref_loss.item():.6f}")
    print(f"Our loss: {our_loss.item():.6f}")
    print(f"Absolute difference: {diff:.8f}")
    
    if diff < 1e-5:
        print("✅ DKD loss functions are consistent!")
    else:
        print("❌ DKD loss functions differ!")
        
    return diff < 1e-5

def test_lskd_properties():
    """Test LSKD mathematical properties after our fixes"""
    print("\n=== Testing LSKD Properties ===")
    
    torch.manual_seed(42)
    batch_size, num_classes = 32, 100
    temperature = 4.0
    
    # Generate test logits
    logits = torch.randn(batch_size, num_classes)
    
    # Apply our normalization 
    normalized = normalize_logit(logits)
    
    # Test properties
    mean_vals = normalized.mean(dim=1)
    std_vals = normalized.std(dim=1)
    
    print(f"Mean of normalized logits: {mean_vals.mean().item():.2e} (should be ~0)")
    print(f"Std of normalized logits: {std_vals.mean().item():.6f} (should be ~1)")
    
    # Test softmax with temperature
    probs = F.softmax(normalized / temperature, dim=1)
    print(f"Softmax probabilities sum: {probs.sum(dim=1).mean().item():.6f} (should be 1)")
    
    # Check range
    min_val = normalized.min().item()
    max_val = normalized.max().item()
    expected_bound = np.sqrt(num_classes - 1)
    print(f"Normalized range: [{min_val:.3f}, {max_val:.3f}]")
    print(f"Expected bound: ±{expected_bound:.3f}")
    
    mean_ok = abs(mean_vals.mean().item()) < 1e-6
    std_ok = abs(std_vals.mean().item() - 1.0) < 1e-6
    
    if mean_ok and std_ok:
        print("✅ LSKD properties verified!")
    else:
        print("❌ LSKD properties not satisfied!")
        
    return mean_ok and std_ok

def main():
    print("Testing LSKD Implementation Consistency")
    print("=" * 50)
    
    # Run all tests
    norm_ok = test_normalization_consistency()
    loss_ok = test_dkd_loss_consistency()
    props_ok = test_lskd_properties()
    
    print("\n" + "=" * 50)
    print("FINAL RESULTS:")
    print(f"Normalization consistency: {'✅' if norm_ok else '❌'}")
    print(f"DKD loss consistency: {'✅' if loss_ok else '❌'}")
    print(f"LSKD properties: {'✅' if props_ok else '❌'}")
    
    if norm_ok and loss_ok and props_ok:
        print("\n🎉 All tests passed! LSKD implementation is consistent!")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
