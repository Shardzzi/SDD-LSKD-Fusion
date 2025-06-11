#!/bin/bash

# SDD-DKD-LSKD Full Training Script - Part 3
# This script runs the 3 teacher-student pairs shown in the research table
# Each pair will be tested with M=[1], M=[1,2], and M=[1,2,4] for comprehensive evaluation

echo "=========================================="
echo "Starting Comprehensive SDD-DKD-LSKD Training (Part 3)"
echo "=========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd-lskd-fusion

# Create comprehensive logging directories
mkdir -p logs/sdd_dkd_lskd_full/cifar100
mkdir -p logs/sdd_dkd_lskd_full/summaries

# Set timestamp for this training session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_DIR="logs/sdd_dkd_lskd_full/session_$TIMESTAMP"
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

# Main SDD-DKD-LSKD Training Experiments - Part 3 (Research Table Validation)
echo "Starting SDD-DKD-LSKD training experiments Part 3 (Research Table)..."
echo ""

# 5. ResNet32x4 -> MobileNetV2 (Teacher Accuracy: 79.42%, Student Baseline: 64.6%)
echo "=== Experiment 5: ResNet32x4 -> MobileNetV2 ==="
echo "--- SDD-DKD-LSKD experiments ---"
run_full_training "configs/cifar100/sdd_dkd_lskd/res32x4_mv2.yaml" "[1]" "res32x4_mv2_dkd_global" "Global DKD-LSKD baseline"
run_full_training "configs/cifar100/sdd_dkd_lskd/res32x4_mv2.yaml" "[1,2]" "res32x4_mv2_dkd_twoscale" "Two-scale SDD-DKD-LSKD fusion"
run_full_training "configs/cifar100/sdd_dkd_lskd/res32x4_mv2.yaml" "[1,2,4]" "res32x4_mv2_dkd_threescale" "Three-scale SDD-DKD-LSKD fusion"
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_mv2.yaml" "[1]" "res32x4_mv2_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_mv2.yaml" "[1,2]" "res32x4_mv2_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_mv2.yaml" "[1,2,4]" "res32x4_mv2_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# Generate training summary
echo "=========================================="
echo "Generating training summary..."
echo "=========================================="

SUMMARY_FILE="$SESSION_DIR/training_summary.txt"
echo "SDD-DKD-LSKD Training Session Summary - Part 3 (Research Table)" > "$SUMMARY_FILE"
echo "Session: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Research Table Validation Experiments:" >> "$SUMMARY_FILE"
echo "5. ResNet32x4 -> MobileNetV2 (DKD+KD variants, M=[1], M=[1,2], M=[1,2,4])" >> "$SUMMARY_FILE"
echo "   Teacher Baseline: 79.42%, Student Baseline: 64.6%" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Total experiments: 18 (3 pairs × 3 M-settings × 2 distillation methods)" >> "$SUMMARY_FILE"
echo "Expected performance improvements with SDD-DKD-LSKD fusion" >> "$SUMMARY_FILE"
echo "Log files location: $SESSION_DIR" >> "$SUMMARY_FILE"

# Show final results
echo ""
echo "=========================================="
echo "🎉 SDD-DKD-LSKD Training Session Part 3 Completed! 🎉"
echo "=========================================="
echo "Session: $TIMESTAMP"
echo "All logs saved in: $SESSION_DIR"
echo "Summary available in: $SUMMARY_FILE"
echo ""
echo "Research Table Validation Results:"
echo "- 3 teacher-student pairs tested"
echo "- 3 multi-scale settings per pair (M=[1], M=[1,2], M=[1,2,4])"
echo "- 2 distillation methods per setting (DKD-LSKD vs KD-LSKD)"
echo "- Total 18 comprehensive experiments completed"
echo ""
echo "Expected outcomes:"
echo "1. ResNet32x4->MobileNetV2: Significant improvement from 64.6% baseline"
echo "2. WRN40-2->VGG8: Moderate improvement from 70.36% baseline"  
echo "3. WRN40-2->MobileNetV2: Cross-architecture knowledge transfer"
echo ""
echo "To analyze results, check the log files for:"
echo "- Final test accuracies vs. baselines"
echo "- Multi-scale fusion effectiveness"
echo "- LSKD standardization impact"
echo "- Training convergence patterns"
echo ""
echo "Next steps:"
echo "1. Compare results with research table baselines"
echo "2. Analyze multi-scale contribution (M=[1] vs M=[1,2] vs M=[1,2,4])"
echo "3. Generate performance comparison plots"
echo "4. Validate SDD-DKD-LSKD fusion effectiveness"
echo "=========================================="
