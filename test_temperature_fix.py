#!/usr/bin/env python3
"""
Test script to verify the temperature scaling fix in SDD-LSKD implementation
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add the mdistiller package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from mdistiller.distillers.SDD_LSKD import normalize_logit, dkd_loss_with_lskd


def test_temperature_consistency():
    """Test that temperature scaling is consistent with Algorithm 2"""
    print("=== Testing Temperature Scaling Fix ===")
    
    # Create sample data
    batch_size = 4
    num_classes = 10
    temperature = 4.0
    
    # Generate sample logits
    torch.manual_seed(42)  # For reproducible results
    logits_student = torch.randn(batch_size, num_classes) * 5
    logits_teacher = torch.randn(batch_size, num_classes) * 5
    target = torch.randint(0, num_classes, (batch_size,))
    
    print(f"Temperature: {temperature}")
    print(f"Original student logits mean: {logits_student.mean():.4f}, std: {logits_student.std():.4f}")
    print(f"Original teacher logits mean: {logits_teacher.mean():.4f}, std: {logits_teacher.std():.4f}")
    
    # Test LSKD implementation
    print("\n--- LSKD Implementation (Algorithm 2) ---")
    loss_lskd = dkd_loss_with_lskd(
        logits_student.clone(), logits_teacher.clone(), target,
        alpha=1.0, beta=8.0, temperature=temperature
    )
    print(f"LSKD loss: {loss_lskd.item():.6f}")
    
    # Compare with traditional KD (we'll simulate this manually)
    print("\n--- Traditional KD Simulation ---")
    pred_s_trad = F.softmax(logits_student / temperature, dim=1)
    pred_t_trad = F.softmax(logits_teacher / temperature, dim=1)
    loss_traditional = F.kl_div(F.log_softmax(logits_student / temperature, dim=1), 
                               pred_t_trad, reduction='batchmean') * (temperature ** 2)
    print(f"Traditional KD loss: {loss_traditional.item():.6f}")
    
    # Test manual verification of Algorithm 2
    print("\n--- Manual Algorithm 2 Verification ---")
    # Step 3: q(vn) ← softmax[Z(vn; τ)]
    standardized_teacher = normalize_logit(logits_teacher.clone(), temperature)
    q_vn = F.softmax(standardized_teacher, dim=1)
    
    # Step 4: q(zn) ← softmax[Z(zn; τ)]  
    standardized_student = normalize_logit(logits_student.clone(), temperature)
    q_zn = F.softmax(standardized_student, dim=1)
    
    print(f"After standardization:")
    print(f"  Teacher logits mean: {standardized_teacher.mean():.4f}, std: {standardized_teacher.std():.4f}")
    print(f"  Student logits mean: {standardized_student.mean():.4f}, std: {standardized_student.std():.4f}")
    print(f"  Teacher softmax sum: {q_vn.sum(dim=1).mean():.4f}")
    print(f"  Student softmax sum: {q_zn.sum(dim=1).mean():.4f}")
    
    # Verify that per-sample means are approximately 0 after standardization
    teacher_sample_means = standardized_teacher.mean(dim=1)
    student_sample_means = standardized_student.mean(dim=1)
    print(f"  Teacher per-sample means: {teacher_sample_means}")
    print(f"  Student per-sample means: {student_sample_means}")
    
    # Check if means are close to zero (within tolerance)
    teacher_mean_close_to_zero = torch.allclose(teacher_sample_means, torch.zeros_like(teacher_sample_means), atol=1e-6)
    student_mean_close_to_zero = torch.allclose(student_sample_means, torch.zeros_like(student_sample_means), atol=1e-6)
    
    print(f"  Teacher means ≈ 0: {teacher_mean_close_to_zero}")
    print(f"  Student means ≈ 0: {student_mean_close_to_zero}")
    
    # Compare losses
    print(f"\n--- Loss Comparison ---")
    print(f"Loss difference (LSKD vs Traditional): {abs(loss_lskd.item() - loss_traditional.item()):.6f}")
    
    # Test different temperatures
    print(f"\n--- Temperature Sensitivity Test ---")
    temperatures = [1.0, 2.0, 4.0, 8.0]
    for temp in temperatures:
        loss_lskd_temp = dkd_loss_with_lskd(
            logits_student.clone(), logits_teacher.clone(), target,
            alpha=1.0, beta=8.0, temperature=temp
        )
        # Traditional KD simulation
        loss_trad_temp = F.kl_div(F.log_softmax(logits_student / temp, dim=1), 
                                 F.softmax(logits_teacher / temp, dim=1), 
                                 reduction='batchmean') * (temp ** 2)
        print(f"T={temp}: LSKD={loss_lskd_temp.item():.6f}, Traditional={loss_trad_temp.item():.6f}")
    
    print("\n✅ SDD-LSKD temperature scaling verification completed!")
    return True


if __name__ == "__main__":
    test_temperature_consistency()
