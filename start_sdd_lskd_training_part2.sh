#!/bin/bash

# SDD-LSKD Full Training Script - Part 2
# This script runs experiments 3 and 4 of comprehensive SDD-LSKD training

echo "=========================================="
echo "Starting Comprehensive SDD-LSKD Training (Part 2)"
echo "=========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd-lskd-fusion

# Create comprehensive logging directories
mkdir -p logs/sdd_lskd_full/cifar100
mkdir -p logs/sdd_lskd_full/summaries

# Set timestamp for this training session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_DIR="logs/sdd_lskd_full/session_$TIMESTAMP"
mkdir -p "$SESSION_DIR"

echo "Training session: $TIMESTAMP"
echo "All logs will be saved in: $SESSION_DIR"
echo ""

# Function to run full training experiment
run_full_training() {
    local config_file=$1
    local M_setting=$2
    local experiment_name=$3
    local description=$4
    
    echo "=========================================="
    echo "Training: $experiment_name"
    echo "Description: $description"
    echo "M setting: $M_setting"
    echo "Config: $config_file"
    echo "=========================================="
    
    local log_file="$SESSION_DIR/${experiment_name}_M${M_setting}.log"
    local start_time=$(date)
    
    echo "Start time: $start_time" | tee "$log_file"
    echo "Configuration: $config_file" | tee -a "$log_file"
    echo "M setting: $M_setting" | tee -a "$log_file"
    echo "=========================================" | tee -a "$log_file"
    
    # Run the training
    python train_origin.py \
        --cfg "$config_file" \
        --gpu 0 \
        --M "$M_setting" \
        2>&1 | tee -a "$log_file"
    
    local end_time=$(date)
    echo "=========================================" | tee -a "$log_file"
    echo "End time: $end_time" | tee -a "$log_file"
    echo "Training completed for: $experiment_name (M=$M_setting)" | tee -a "$log_file"
    echo ""
}

# Main SDD-LSKD Training Experiments - Part 2
echo "Starting SDD-LSKD training experiments Part 2..."
echo ""

# 3. ResNet32x4 -> ShuffleNetV1 (Heterogeneous architectures)
echo "=== Experiment 3: ResNet32x4 -> ShuffleNetV1 (Heterogeneous) ==="
run_full_training "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1]" "res32x4_shuv1_global" "Global LSKD baseline"
run_full_training "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1,2]" "res32x4_shuv1_twoscale" "Two-scale SDD-LSKD fusion"
run_full_training "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1,2,4]" "res32x4_shuv1_threescale" "Three-scale SDD-LSKD fusion"

# 4. WideResNet-40-2 -> ShuffleNetV1 (WideResNet to lightweight)
echo "=== Experiment 4: WideResNet-40-2 -> ShuffleNetV1 ==="
run_full_training "configs/cifar100/sdd_lskd/wrn40_2_shuv1.yaml" "[1]" "wrn40_2_shuv1_global" "Global LSKD baseline"
run_full_training "configs/cifar100/sdd_lskd/wrn40_2_shuv1.yaml" "[1,2]" "wrn40_2_shuv1_twoscale" "Two-scale SDD-LSKD fusion"
run_full_training "configs/cifar100/sdd_lskd/wrn40_2_shuv1.yaml" "[1,2,4]" "wrn40_2_shuv1_threescale" "Three-scale SDD-LSKD fusion"

# Generate training summary
echo "=========================================="
echo "Generating training summary..."
echo "=========================================="

SUMMARY_FILE="$SESSION_DIR/training_summary.txt"
echo "SDD-LSKD Training Session Summary - Part 2" > "$SUMMARY_FILE"
echo "Session: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Experiments completed:" >> "$SUMMARY_FILE"
echo "3. ResNet32x4 -> ShuffleNetV1 (M=[1], M=[1,2], M=[1,2,4])" >> "$SUMMARY_FILE"
echo "4. WideResNet-40-2 -> ShuffleNetV1 (M=[1], M=[1,2], M=[1,2,4])" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Total experiments: 6" >> "$SUMMARY_FILE"
echo "Log files location: $SESSION_DIR" >> "$SUMMARY_FILE"

# Show final results
echo ""
echo "=========================================="
echo "🎉 SDD-LSKD Training Session Part 2 Completed! 🎉"
echo "=========================================="
echo "Session: $TIMESTAMP"
echo "All logs saved in: $SESSION_DIR"
echo "Summary available in: $SUMMARY_FILE"
echo ""
echo "To analyze results, check the log files for:"
echo "- Final test accuracies"
echo "- Training convergence curves"
echo "- Loss values and distillation metrics"
echo ""
echo "Next steps:"
echo "1. Run analysis scripts to compare results"
echo "2. Generate performance plots"
echo "3. Create evaluation reports"
echo "=========================================="
