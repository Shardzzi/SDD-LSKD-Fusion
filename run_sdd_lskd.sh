#!/bin/bash

# SDD-LSKD Fusion Experiments
# This script runs experiments for the SDD-LSKD fusion method on different datasets and architectures

echo "Starting SDD-LSKD Fusion Experiments..."
echo "================================================"

# Set conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd

# Create log directory if it doesn't exist
mkdir -p logs/sdd_lskd

# Function to run experiment with logging
run_experiment() {
    local config_file=$1
    local M_setting=$2
    local gpu_id=$3
    local experiment_name=$4
    
    echo "Running: $experiment_name with M=$M_setting"
    local log_file="logs/sdd_lskd/${experiment_name}_M${M_setting}_$(date +%Y%m%d_%H%M%S).log"
    
    python train_origin.py --cfg "$config_file" --gpu "$gpu_id" --M "$M_setting" 2>&1 | tee "$log_file"
    
    echo "Experiment completed. Log saved to: $log_file"
    echo "------------------------------------------------"
}

# CIFAR-100 Experiments
echo "=== CIFAR-100 SDD-LSKD Experiments ==="

# ResNet32x4 -> ResNet8x4 (homogeneous architecture)
run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1]" 0 "cifar100_res32x4_res8x4"
run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1,2]" 0 "cifar100_res32x4_res8x4"
run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1,2,4]" 0 "cifar100_res32x4_res8x4"

# ResNet32x4 -> MobileNetV2 (heterogeneous architecture)
run_experiment "configs/cifar100/sdd_lskd/res32x4_mv2.yaml" "[1]" 0 "cifar100_res32x4_mv2"
run_experiment "configs/cifar100/sdd_lskd/res32x4_mv2.yaml" "[1,2]" 0 "cifar100_res32x4_mv2"
run_experiment "configs/cifar100/sdd_lskd/res32x4_mv2.yaml" "[1,2,4]" 0 "cifar100_res32x4_mv2"

# ResNet32x4 -> ShuffleNetV1 (heterogeneous architecture)
run_experiment "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1]" 0 "cifar100_res32x4_shuv1"
run_experiment "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1,2]" 0 "cifar100_res32x4_shuv1"
run_experiment "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1,2,4]" 0 "cifar100_res32x4_shuv1"

echo "=== CUB-200 SDD-LSKD Experiments ==="

# Fine-grained classification experiments on CUB-200
# ResNet32x4 -> ShuffleNetV1
run_experiment "configs/cub200/sdd_lskd/res32x4_shuv1.yaml" "[1]" 0 "cub200_res32x4_shuv1"
run_experiment "configs/cub200/sdd_lskd/res32x4_shuv1.yaml" "[1,2]" 0 "cub200_res32x4_shuv1"
run_experiment "configs/cub200/sdd_lskd/res32x4_shuv1.yaml" "[1,2,4]" 0 "cub200_res32x4_shuv1"

# ResNet32x4 -> MobileNetV2
run_experiment "configs/cub200/sdd_lskd/res32x4_mv2.yaml" "[1]" 0 "cub200_res32x4_mv2"
run_experiment "configs/cub200/sdd_lskd/res32x4_mv2.yaml" "[1,2]" 0 "cub200_res32x4_mv2"
run_experiment "configs/cub200/sdd_lskd/res32x4_mv2.yaml" "[1,2,4]" 0 "cub200_res32x4_mv2"

echo "================================================"
echo "All SDD-LSKD fusion experiments completed!"
echo "Check logs in logs/sdd_lskd/ for detailed results"
echo "================================================"

# Function to run quick comparison experiments (shorter epochs for validation)
run_quick_experiment() {
    local config_file=$1
    local M_setting=$2
    local gpu_id=$3
    local experiment_name=$4
    
    echo "Running QUICK: $experiment_name with M=$M_setting (10 epochs)"
    local log_file="logs/sdd_lskd/quick_${experiment_name}_M${M_setting}_$(date +%Y%m%d_%H%M%S).log"
    
    # Modify epochs to 10 for quick testing
    python train_origin.py --cfg "$config_file" --gpu "$gpu_id" --M "$M_setting" --epochs 10 2>&1 | tee "$log_file"
    
    echo "Quick experiment completed. Log saved to: $log_file"
    echo "------------------------------------------------"
}

# Quick validation function
run_quick_validation() {
    echo ""
    echo "=== Quick Validation Experiments (10 epochs each) ==="
    echo "These experiments run for only 10 epochs to quickly validate the implementation"
    
    # Test one configuration for each M setting
    run_quick_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1]" 0 "quick_validation_global"
    run_quick_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1,2]" 0 "quick_validation_multiscale2"
    run_quick_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1,2,4]" 0 "quick_validation_multiscale3"
    
    echo "Quick validation completed!"
}

# Uncomment the line below to run quick validation instead of full experiments
# run_quick_validation
