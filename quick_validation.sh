#!/bin/bash

# SDD-LSKD Quick Test Script
# This script runs a minimal set of experiments to validate the fusion method

echo "SDD-LSKD Quick Validation Test"
echo "=============================="

# Configuration
CONDA_ENV="sdd-lskd-fusion"
GPU_ID=0

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
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Create log directory
mkdir -p logs/quick_validation

# Run quick tests
echo ""
echo "Running quick validation tests (5 epochs each)..."
echo "=================================================="

# Test 1: Global distillation (M=[1])
echo "Test 1: Global LSKD (M=[1])"
python train_origin.py \
    --cfg configs/cifar100/sdd_lskd/quick_test.yaml \
    --gpu $GPU_ID \
    --M "[1]" \
    2>&1 | tee logs/quick_validation/test1_global_lskd.log

echo ""

# Test 2: Two-scale fusion (M=[1,2])
echo "Test 2: Two-scale SDD-LSKD (M=[1,2])"
python train_origin.py \
    --cfg configs/cifar100/sdd_lskd/quick_test.yaml \
    --gpu $GPU_ID \
    --M "[1,2]" \
    2>&1 | tee logs/quick_validation/test2_twoscale_fusion.log

echo ""

# Test 3: Three-scale fusion (M=[1,2,4])
echo "Test 3: Three-scale SDD-LSKD (M=[1,2,4])"
python train_origin.py \
    --cfg configs/cifar100/sdd_lskd/quick_test.yaml \
    --gpu $GPU_ID \
    --M "[1,2,4]" \
    2>&1 | tee logs/quick_validation/test3_threescale_fusion.log

echo ""
echo "Quick validation completed!"
echo "=========================="

# Extract and display results
echo "Results Summary:"
echo "==============="

for i in 1 2 3; do
    case $i in
        1) test_name="Global LSKD (M=[1])" ;;
        2) test_name="Two-scale SDD-LSKD (M=[1,2])" ;;
        3) test_name="Three-scale SDD-LSKD (M=[1,2,4])" ;;
    esac
    
    log_file="logs/quick_validation/test${i}_*.log"
    if ls $log_file 1> /dev/null 2>&1; then
        final_acc=$(tail -20 $log_file | grep -oE "Test Acc: [0-9]+\.[0-9]+" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
        if [ -n "$final_acc" ]; then
            echo "Test $i - $test_name: $final_acc%"
        else
            echo "Test $i - $test_name: No accuracy found (check log)"
        fi
    else
        echo "Test $i - $test_name: Log file not found"
    fi
done

echo ""
echo "All logs saved in: logs/quick_validation/"
echo "If all tests completed without errors, SDD-LSKD fusion is working correctly!"
