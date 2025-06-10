#!/bin/bash

# SDD-LSKD Quick Test Script
# Runs short experiments to validate the fusion method works correctly

echo "SDD-LSKD Quick Validation Tests"
echo "==============================="

# Set conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd

# Create log directory
mkdir -p logs/sdd_lskd_quick

# Quick test function - uses quick config with 5 epochs
run_quick_test() {
    local M_setting=$1
    local test_name=$2
    
    echo "Testing: $test_name with M=$M_setting (5 epochs)"
    local log_file="logs/sdd_lskd_quick/${test_name}_M${M_setting//[,\[\]]/}_$(date +%H%M%S).log"
    
    # Run with quick config (5 epochs)
    timeout 600 python train_origin.py --cfg "configs/cifar100/sdd_lskd/quick_test.yaml" --gpu 0 --M "$M_setting" 2>&1 | tee "$log_file"
    
    echo "Test completed. Log: $log_file"
    echo "---"
}

echo "Testing SDD-LSKD fusion method with different multi-scale settings..."

# Test 1: Global distillation (M=[1]) - should use LSKD standardization only
run_quick_test "[1]" "global_lskd"

# Test 2: Two-scale distillation (M=[1,2]) - should use SDD + LSKD
run_quick_test "[1,2]" "twoscale_sdd_lskd"

# Test 3: Three-scale distillation (M=[1,2,4]) - full SDD + LSKD fusion
run_quick_test "[1,2,4]" "threescale_sdd_lskd"

echo ""
echo "Quick validation tests completed!"
echo "Check logs in logs/sdd_lskd_quick/ for results"
echo ""
echo "If all tests run without errors, the SDD-LSKD fusion is working correctly!"
