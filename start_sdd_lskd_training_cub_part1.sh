#!/bin/bash

# SDD-KD-LSKD CUB-200-2011 Training Script - Part 1
# This script runs experiments 1-4 of comprehensive SDD-KD-LSKD training on CUB-200-2011 dataset

echo "=========================================="
echo "Starting Comprehensive SDD-KD-LSKD Training (CUB-200-2011 Part 1)"
echo "=========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd-lskd-fusion

# Create comprehensive logging directories
mkdir -p logs/sdd_kd_lskd_full/cub200
mkdir -p logs/sdd_kd_lskd_full/summaries

# Set timestamp for this training session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_DIR="logs/sdd_kd_lskd_full/cub200/session_$TIMESTAMP"
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

# Main SDD-KD-LSKD Training Experiments - CUB-200-2011 Part 1
echo "Starting main SDD-KD-LSKD training experiments (CUB-200-2011)..."
echo ""

# 1. ResNet32x4 -> ResNet8x4 (Primary experiment - homogeneous architectures)
echo "=== Experiment 1: ResNet32x4 -> ResNet8x4 (Homogeneous) ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_res8x4.yaml" "[1]" "res32x4_res8x4_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_res8x4.yaml" "[1,2]" "res32x4_res8x4_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_res8x4.yaml" "[1,2,4]" "res32x4_res8x4_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# 2. ResNet32x4 -> ShuffleNetV1 (Heterogeneous architectures)
echo "=== Experiment 2: ResNet32x4 -> ShuffleNetV1 (Heterogeneous) ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_shuv1.yaml" "[1]" "res32x4_shuv1_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_shuv1.yaml" "[1,2]" "res32x4_shuv1_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_shuv1.yaml" "[1,2,4]" "res32x4_shuv1_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# 3. ResNet32x4 -> MobileNetV2 (ResNet to MobileNet)
echo "=== Experiment 3: ResNet32x4 -> MobileNetV2 ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_mv2.yaml" "[1]" "res32x4_mv2_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_mv2.yaml" "[1,2]" "res32x4_mv2_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cub200/sdd_kd_lskd/res32x4_mv2.yaml" "[1,2,4]" "res32x4_mv2_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# 4. WideResNet-40-2 -> ShuffleNetV1 (WideResNet to lightweight)
echo "=== Experiment 4: WideResNet-40-2 -> ShuffleNetV1 ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cub200/sdd_kd_lskd/wrn40_2_shuv1.yaml" "[1]" "wrn40_2_shuv1_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cub200/sdd_kd_lskd/wrn40_2_shuv1.yaml" "[1,2]" "wrn40_2_shuv1_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cub200/sdd_kd_lskd/wrn40_2_shuv1.yaml" "[1,2,4]" "wrn40_2_shuv1_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# Generate training summary
echo "=========================================="
echo "Generating training summary..."
echo "=========================================="

SUMMARY_FILE="$SESSION_DIR/training_summary.txt"
echo "SDD-KD-LSKD Training Session Summary - CUB-200-2011 Part 1" > "$SUMMARY_FILE"
echo "Session: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Experiments completed:" >> "$SUMMARY_FILE"
echo "1. ResNet32x4 -> ResNet8x4 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "2. ResNet32x4 -> ShuffleNetV1 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "3. ResNet32x4 -> MobileNetV2 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "4. WideResNet-40-2 -> ShuffleNetV1 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Total experiments: 12 (4 pairs × 3 M-settings)" >> "$SUMMARY_FILE"
echo "Dataset: CUB-200-2011 (200 bird species)" >> "$SUMMARY_FILE"
echo "Log files location: $SESSION_DIR" >> "$SUMMARY_FILE"

# Show final results
echo ""
echo "=========================================="
echo "🎉 SDD-KD-LSKD Training Session Part 1 (CUB-200-2011) Completed! 🎉"
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
echo "1. Run Part 2 for research table validation experiments"
echo "2. Run analysis scripts to compare results"
echo "3. Generate performance plots"
echo "4. Create evaluation reports"
echo "=========================================="
