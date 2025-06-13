#!/bin/bash

# Test Script for SDD Ablation Study (Experiment 2)
# Testing M=[1,4] settings for two teacher-student pairs

echo "=========================================="
echo "Testing SDD Ablation Study (Experiment 2)"
echo "=========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd-lskd-fusion

# Create test logging directories
mkdir -p logs/sdd_kd_lskd_full/cifar100/test_ablation
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_DIR="logs/sdd_kd_lskd_full/cifar100/test_ablation/session_$TIMESTAMP"
mkdir -p "$SESSION_DIR"

echo "Test session: $TIMESTAMP"
echo "Test logs will be saved in: $SESSION_DIR"
echo ""

# Function to run test training experiment (short version)
run_test_training() {
    local config_file=$1
    local M_setting=$2
    local experiment_name=$3
    local description=$4
    
    echo "=========================================="
    echo "Testing: $experiment_name"
    echo "Description: $description"
    echo "M setting: $M_setting"
    echo "Config: $config_file"
    echo "=========================================="
    
    local log_file="$SESSION_DIR/${experiment_name}_M${M_setting}_test.log"
    local start_time=$(date)
    
    echo "Start time: $start_time" | tee "$log_file"
    echo "Configuration: $config_file" | tee -a "$log_file"
    echo "M setting: $M_setting" | tee -a "$log_file"
    echo "=========================================" | tee -a "$log_file"
    
    # Run a short test training (only 1 epoch for quick validation)
    python train_origin.py \
        --cfg "$config_file" \
        --gpu 0 \
        --M "$M_setting" \
        --epochs 1 \
        2>&1 | tee -a "$log_file"
    
    local end_time=$(date)
    local exit_status=$?
    echo "=========================================" | tee -a "$log_file"
    echo "End time: $end_time" | tee -a "$log_file"
    echo "Exit status: $exit_status" | tee -a "$log_file"
    echo "Test completed for: $experiment_name (M=$M_setting)" | tee -a "$log_file"
    echo ""
    
    return $exit_status
}

echo "########################################"
echo "SDD Ablation Study Test"
echo "########################################"
echo ""

# Test the exact code from the training script
echo "=== Experiment 2: SDD Ablation Study ==="
echo "--- KD-LSKD ablation ---"

# Test 1: ResNet32x4->ResNet8x4 with M=[1,4]
echo "Testing ResNet32x4->ResNet8x4 with M=[1,4]..."
run_test_training "configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml" "[1,4]" "res32x4_res8x4_kd_scale1_4" "ResNet32x4->ResNet8x4 KD M=[1,4] ablation"
TEST1_STATUS=$?

# Test 2: ResNet32x4->ShuffleNetV1 with M=[1,4]
echo "Testing ResNet32x4->ShuffleNetV1 with M=[1,4]..."
run_test_training "configs/cifar100/sdd_kd_lskd/res32x4_shuv1.yaml" "[1,4]" "res32x4_shuv1_kd_scale1_4" "ResNet32x4->ShuffleNetV1 KD M=[1,4] ablation"
TEST2_STATUS=$?

# Summary
echo "=========================================="
echo "SDD Ablation Study Test Results"
echo "=========================================="
echo "Test session: $TIMESTAMP"
echo "Test logs location: $SESSION_DIR"
echo ""
echo "Test Results:"
echo "1. ResNet32x4->ResNet8x4 M=[1,4]: $([ $TEST1_STATUS -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED (exit code: $TEST1_STATUS)")"
echo "2. ResNet32x4->ShuffleNetV1 M=[1,4]: $([ $TEST2_STATUS -eq 0 ] && echo "✅ PASSED" || echo "❌ FAILED (exit code: $TEST2_STATUS)")"
echo ""

if [ $TEST1_STATUS -eq 0 ] && [ $TEST2_STATUS -eq 0 ]; then
    echo "🎉 All SDD Ablation Study tests PASSED!"
    echo "The code segment should work properly in the full training script."
else
    echo "⚠️  Some tests FAILED. Check the log files for details:"
    echo "   - Log directory: $SESSION_DIR"
    echo "   - Check configuration files and dependencies"
fi

echo "=========================================="
