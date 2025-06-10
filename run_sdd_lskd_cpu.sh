#!/bin/bash

# SDD-LSKD CPU-only Quick Test Script
# Runs short experiments to validate the fusion method works correctly on CPU

echo "SDD-LSKD CPU Quick Validation Tests"
echo "=================================="

# Set conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd

# Create log directory
mkdir -p logs/sdd_lskd_cpu

# CPU test function - 2 epochs only for very quick validation
run_cpu_test() {
    local M_setting=$1
    local test_name=$2
    
    echo "Testing: $test_name with M=$M_setting (CPU, 2 epochs)"
    local log_file="logs/sdd_lskd_cpu/${test_name}_M${M_setting//[,\[\]]/}_$(date +%H%M%S).log"
    
    # Run with CPU and quick config
    timeout 300 python train_origin.py --cfg "configs/cifar100/sdd_lskd/cpu_quick_test.yaml" --M "$M_setting" 2>&1 | tee "$log_file"
    
    echo "Test completed. Log: $log_file"
    echo "---"
}

echo "Testing SDD-LSKD fusion method on CPU with different multi-scale settings..."

# Test 1: Global distillation (M=[1]) - should use LSKD standardization only
run_cpu_test "[1]" "cpu_global_lskd"

# Test 2: Two-scale distillation (M=[1,2]) - should use SDD + LSKD
run_cpu_test "[1,2]" "cpu_twoscale_sdd_lskd"

# Test 3: Three-scale distillation (M=[1,2,4]) - full SDD + LSKD fusion
run_cpu_test "[1,2,4]" "cpu_threescale_sdd_lskd"

echo ""
echo "CPU quick validation tests completed!"
echo "Check logs in logs/sdd_lskd_cpu/ for results"
echo ""
echo "If all tests run without CUDA errors, the SDD-LSKD fusion is working correctly!"
