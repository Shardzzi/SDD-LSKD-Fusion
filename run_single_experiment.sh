#!/bin/bash

# Single SDD-LSKD Experiment Runner
# Usage: ./run_single_experiment.sh <config_file> <M_setting> [experiment_name] [gpu_id]

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <config_file> <M_setting> [experiment_name] [gpu_id]"
    echo ""
    echo "Examples:"
    echo "  $0 configs/cifar100/sdd_lskd/res32x4_res8x4.yaml \"[1,2,4]\" \"test_experiment\" 0"
    echo "  $0 configs/cifar100/sdd_lskd/quick_test.yaml \"[1]\" \"quick_test\""
    echo ""
    echo "Available configs:"
    echo "  - configs/cifar100/sdd_lskd/quick_test.yaml (5 epochs)"
    echo "  - configs/cifar100/sdd_lskd/res32x4_res8x4.yaml (240 epochs)"
    echo "  - configs/cifar100/sdd_lskd/res32x4_mv2.yaml (240 epochs)"
    echo "  - configs/cifar100/sdd_lskd/res32x4_shuv1.yaml (240 epochs)"
    echo ""
    echo "M_setting examples:"
    echo "  - \"[1]\" - Global distillation with LSKD standardization"
    echo "  - \"[1,2]\" - Two-scale SDD + LSKD fusion"
    echo "  - \"[1,2,4]\" - Three-scale SDD + LSKD fusion"
    exit 1
fi

# Parse arguments
CONFIG_FILE=$1
M_SETTING=$2
EXPERIMENT_NAME=${3:-"experiment_$(date +%H%M%S)"}
GPU_ID=${4:-0}

# Configuration
CONDA_ENV="sdd-lskd-fusion"

echo "SDD-LSKD Single Experiment Runner"
echo "================================="
echo "Config file: $CONFIG_FILE"
echo "M setting: $M_SETTING"
echo "Experiment name: $EXPERIMENT_NAME"
echo "GPU ID: $GPU_ID"
echo ""

# Validate config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Setup environment
echo "Setting up environment..."
source /root/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate conda environment: $CONDA_ENV"
    exit 1
fi

echo "Environment activated successfully"
python --version

# Create log directory
LOG_DIR="logs/single_experiments"
mkdir -p $LOG_DIR

# Generate log file name
SANITIZED_M=$(echo "$M_SETTING" | tr -d '[],' | tr ' ' '_')
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/${EXPERIMENT_NAME}_M${SANITIZED_M}_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo ""

# Run the experiment
echo "Starting experiment..."
echo "====================="

# Start timing
start_time=$(date +%s)

# Run with timeout (2 hours for full experiments, 30 minutes for quick tests)
if [[ "$CONFIG_FILE" == *"quick_test"* ]]; then
    TIMEOUT=1800  # 30 minutes for quick tests
else
    TIMEOUT=7200  # 2 hours for full experiments
fi

echo "Timeout set to: $TIMEOUT seconds"

if timeout $TIMEOUT python train_origin.py \
    --cfg "$CONFIG_FILE" \
    --gpu $GPU_ID \
    --M "$M_SETTING" \
    2>&1 | tee "$LOG_FILE"; then
    
    # Calculate duration
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    hours=$((duration / 3600))
    minutes=$(((duration % 3600) / 60))
    seconds=$((duration % 60))
    
    echo ""
    echo "Experiment completed successfully!"
    echo "Duration: ${hours}h ${minutes}m ${seconds}s"
    
    # Extract final accuracy
    final_acc=$(tail -20 "$LOG_FILE" | grep -oE "Test Acc: [0-9]+\.[0-9]+" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
    if [ -n "$final_acc" ]; then
        echo "Final Test Accuracy: $final_acc%"
        echo "FINAL_ACCURACY: $final_acc%" >> "$LOG_FILE"
    else
        echo "Warning: Could not extract final accuracy from log"
    fi
    
    echo "Log saved to: $LOG_FILE"
    
else
    echo ""
    echo "Error: Experiment failed or timed out"
    echo "EXPERIMENT_FAILED_OR_TIMEOUT" >> "$LOG_FILE"
    exit 1
fi

echo ""
echo "Experiment summary:"
echo "=================="
echo "Config: $CONFIG_FILE"
echo "M setting: $M_SETTING"
echo "Duration: ${hours}h ${minutes}m ${seconds}s"
if [ -n "$final_acc" ]; then
    echo "Final accuracy: $final_acc%"
fi
echo "Log file: $LOG_FILE"
