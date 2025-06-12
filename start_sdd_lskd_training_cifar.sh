#!/bin/bash

# SDD-KD-LSKD CIFAR-100 Complete Training Script
# This script runs comprehensive SDD-KD-LSKD training experiments on CIFAR-100 dataset
# Combines all experiments from part1, part2, and part3

echo "=========================================="
echo "Starting Comprehensive SDD-KD-LSKD Training (CIFAR-100)"
echo "=========================================="

# Set up environment
export CUDA_VISIBLE_DEVICES=0
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sdd-lskd-fusion

# Create comprehensive logging directories
mkdir -p logs/sdd_kd_lskd_full/cifar100
mkdir -p logs/sdd_kd_lskd_full/summaries

# Set timestamp for this training session
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_DIR="logs/sdd_kd_lskd_full/cifar100/session_$TIMESTAMP"
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

# ================================================================================
# PART 1: Main SDD-KD-LSKD Training Experiments
# ================================================================================

echo "########################################"
echo "PART 1: Main SDD-KD-LSKD Training Experiments"
echo "########################################"
echo ""

# 1. ResNet32x4 -> ResNet8x4 (Primary experiment - homogeneous architectures)
echo "=== Experiment 1: ResNet32x4 -> ResNet8x4 (Homogeneous) ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml" "[1]" "res32x4_res8x4_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml" "[1,2]" "res32x4_res8x4_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml" "[1,2,4]" "res32x4_res8x4_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# 2. SDD Ablation Study
echo "=== Experiment 2: SDD Ablation Study ==="
echo "--- KD-LSKD ablation ---"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_res8x4.yaml" "[1,4]" "res32x4_res8x4_kd_scale1_4" "ResNet32x4->ResNet8x4 KD M=[1,4] ablation"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_shuv1.yaml" "[1,4]" "res32x4_shuv1_kd_scale1_4" "ResNet32x4->ShuffleNetV1 KD M=[1,4] ablation"

echo ""
echo "PART 1 completed. Moving to PART 2..."
echo ""

# ================================================================================
# PART 2: Heterogeneous Architecture Experiments
# ================================================================================

echo "########################################"
echo "PART 2: Heterogeneous Architecture Experiments"
echo "########################################"
echo ""

# 3. ResNet32x4 -> ShuffleNetV1 (Heterogeneous architectures)
echo "=== Experiment 3: ResNet32x4 -> ShuffleNetV1 (Heterogeneous) ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_shuv1.yaml" "[1]" "res32x4_shuv1_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_shuv1.yaml" "[1,2]" "res32x4_shuv1_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_shuv1.yaml" "[1,2,4]" "res32x4_shuv1_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# 4. WideResNet-40-2 -> ShuffleNetV1 (WideResNet to lightweight)
echo "=== Experiment 4: WideResNet-40-2 -> ShuffleNetV1 ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_shuv1.yaml" "[1]" "wrn40_2_shuv1_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_shuv1.yaml" "[1,2]" "wrn40_2_shuv1_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_shuv1.yaml" "[1,2,4]" "wrn40_2_shuv1_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

echo ""
echo "PART 2 completed. Moving to PART 3..."
echo ""

# ================================================================================
# PART 3: Research Table Validation Experiments
# ================================================================================

echo "########################################"
echo "PART 3: Research Table Validation Experiments"
echo "########################################"
echo ""

# 5. ResNet32x4 -> MobileNetV2 (Teacher Accuracy: 79.42%, Student Baseline: 64.6%)
echo "=== Experiment 5: ResNet32x4 -> MobileNetV2 ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_mv2.yaml" "[1]" "res32x4_mv2_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_mv2.yaml" "[1,2]" "res32x4_mv2_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cifar100/sdd_kd_lskd/res32x4_mv2.yaml" "[1,2,4]" "res32x4_mv2_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# 6. WRN40-2 -> VGG8 (Teacher Accuracy: 75.61%, Student Baseline: 70.36%)
echo "=== Experiment 6: WRN40-2 -> VGG8 ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_vgg8.yaml" "[1]" "wrn40_2_vgg8_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_vgg8.yaml" "[1,2]" "wrn40_2_vgg8_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_vgg8.yaml" "[1,2,4]" "wrn40_2_vgg8_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# 7. WRN40-2 -> MobileNetV2 (Teacher Accuracy: 75.61%, Student Baseline: 64.6%)
echo "=== Experiment 7: WRN40-2 -> MobileNetV2 ==="
echo "--- SDD-KD-LSKD experiments ---"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_mv2.yaml" "[1]" "wrn40_2_mv2_kd_global" "Global KD-LSKD baseline"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_mv2.yaml" "[1,2]" "wrn40_2_mv2_kd_twoscale" "Two-scale SDD-KD-LSKD fusion"
run_full_training "configs/cifar100/sdd_kd_lskd/wrn40_2_mv2.yaml" "[1,2,4]" "wrn40_2_mv2_kd_threescale" "Three-scale SDD-KD-LSKD fusion"

# ================================================================================
# Generate comprehensive training summary
# ================================================================================

echo "=========================================="
echo "Generating comprehensive training summary..."
echo "=========================================="

SUMMARY_FILE="$SESSION_DIR/training_summary.txt"
echo "SDD-KD-LSKD Complete Training Session Summary - CIFAR-100" > "$SUMMARY_FILE"
echo "Session: $TIMESTAMP" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "=========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "PART 1 - Main Experiments:" >> "$SUMMARY_FILE"
echo "1. ResNet32x4 -> ResNet8x4 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "2. SDD Ablation Study - SDD-KD-LSKD (M=[1,4])" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "PART 2 - Heterogeneous Architectures:" >> "$SUMMARY_FILE"
echo "3. ResNet32x4 -> ShuffleNetV1 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "4. WideResNet-40-2 -> ShuffleNetV1 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "PART 3 - Research Table Validation:" >> "$SUMMARY_FILE"
echo "5. ResNet32x4 -> MobileNetV2 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "6. WRN40-2 -> VGG8 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "7. WRN40-2 -> MobileNetV2 (M=[1], M=[1,2], M=[1,2,4]) - SDD-KD-LSKD" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"
echo "Total experiments: 20 (7 pairs with various M-settings)" >> "$SUMMARY_FILE"
echo "Dataset: CIFAR-100 (100 classes)" >> "$SUMMARY_FILE"
echo "Log files location: $SESSION_DIR" >> "$SUMMARY_FILE"

# Show final results
echo ""
echo "=========================================="
echo "🎉 Complete SDD-KD-LSKD Training Session (CIFAR-100) Completed! 🎉"
echo "=========================================="
echo "Session: $TIMESTAMP"
echo "All logs saved in: $SESSION_DIR"
echo "Summary available in: $SUMMARY_FILE"
echo ""
echo "Training Results Summary:"
echo "- PART 1: 5 experiments (main + ablation)"
echo "- PART 2: 6 experiments (heterogeneous architectures)"
echo "- PART 3: 9 experiments (research table validation)"
echo "- Total: 20 comprehensive experiments completed"
echo ""
echo "Expected outcomes:"
echo "1. Baseline performance establishment"
echo "2. Multi-scale fusion effectiveness validation"
echo "3. Cross-architecture knowledge transfer analysis"
echo "4. Research table results reproduction"
echo ""
echo "To analyze results, check the log files for:"
echo "- Final test accuracies vs. baselines"
echo "- Multi-scale fusion effectiveness"
echo "- LSKD standardization impact"
echo "- Training convergence patterns"
echo ""
echo "Next steps:"
echo "1. Run analysis scripts to compare results"
echo "2. Generate performance comparison plots"
echo "3. Create comprehensive evaluation reports"
echo "4. Validate SDD-KD-LSKD fusion effectiveness"
echo "=========================================="
