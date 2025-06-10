#!/usr/bin/env python3
"""
Test script to verify the tensor conversion fix
"""

import torch
import torch.nn.functional as F

# Simulate the error scenario
def test_tensor_conversion():
    # Create a tensor with 64 elements (batch size)
    batch_size = 64
    loss_tensor = torch.randn(batch_size)
    
    print(f"Loss tensor shape: {loss_tensor.shape}")
    print(f"Loss tensor elements: {loss_tensor.numel()}")
    
    try:
        # This should fail with the original error
        scalar_value = loss_tensor.item()
        print("ERROR: This should have failed!")
    except RuntimeError as e:
        print(f"Expected error: {e}")
    
    # This should work - the fix
    scalar_value = loss_tensor.mean().item()
    print(f"Fixed: Using .mean().item() = {scalar_value}")
    
    # Test when tensor is already scalar
    scalar_tensor = torch.tensor(5.0)
    print(f"Scalar tensor: {scalar_tensor.item()}")

if __name__ == "__main__":
    test_tensor_conversion()
