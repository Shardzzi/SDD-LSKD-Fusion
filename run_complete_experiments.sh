#!/bin/bash

# SDD-LSKD Complete Experimental Suite
# This script provides a comprehensive set of experiments for the SDD-LSKD fusion method
# Author: Assistant
# Date: June 2025

set -e  # Exit on any error

echo "========================================"
echo "SDD-LSKD Complete Experimental Suite"
echo "========================================"
echo "Current time: $(date)"
echo ""

# Configuration
CONDA_ENV="sdd-lskd-fusion"
BASE_LOG_DIR="logs"
EXPERIMENT_SET="full"  # Options: quick, full, ablation, comparison
GPU_ID=0
TIMEOUT_DURATION=7200  # 2 hours per experiment

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to setup environment
setup_environment() {
    print_status "Setting up conda environment..."
    
    # Initialize conda
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
    elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
        source /root/miniconda3/etc/profile.d/conda.sh
    else
        print_error "Conda not found. Please ensure conda is installed."
        exit 1
    fi
    
    # Activate environment
    conda activate $CONDA_ENV
    if [ $? -ne 0 ]; then
        print_error "Failed to activate conda environment: $CONDA_ENV"
        exit 1
    fi
    
    print_success "Environment activated: $CONDA_ENV"
    python --version
    
    # Check required packages
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
}

# Function to create log directories
setup_logging() {
    print_status "Setting up logging directories..."
    
    mkdir -p $BASE_LOG_DIR/sdd_lskd_full
    mkdir -p $BASE_LOG_DIR/sdd_lskd_quick
    mkdir -p $BASE_LOG_DIR/sdd_lskd_ablation
    mkdir -p $BASE_LOG_DIR/sdd_lskd_comparison
    mkdir -p $BASE_LOG_DIR/summaries
    
    print_success "Log directories created"
}

# Function to run a single experiment
run_experiment() {
    local config_file=$1
    local M_setting=$2
    local experiment_name=$3
    local dataset=$4
    local exp_type=${5:-"full"}
    
    local sanitized_M=$(echo "$M_setting" | tr -d '[],' | tr ' ' '_')
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local log_file="$BASE_LOG_DIR/sdd_lskd_${exp_type}/${dataset}_${experiment_name}_M${sanitized_M}_${timestamp}.log"
    
    print_status "Starting experiment: $experiment_name"
    echo "  Config: $config_file"
    echo "  M setting: $M_setting"
    echo "  Dataset: $dataset"
    echo "  Log file: $log_file"
    echo ""
    
    # Run the experiment with timeout
    if timeout $TIMEOUT_DURATION python train_origin.py \
        --cfg "$config_file" \
        --gpu $GPU_ID \
        --M "$M_setting" \
        2>&1 | tee "$log_file"; then
        
        print_success "Experiment completed: $experiment_name"
        
        # Extract final accuracy from log
        local final_acc=$(tail -20 "$log_file" | grep -oE "Test Acc: [0-9]+\.[0-9]+" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
        if [ -n "$final_acc" ]; then
            echo "Final Test Accuracy: $final_acc%" >> "$log_file"
            print_success "Final accuracy: $final_acc%"
        fi
    else
        print_error "Experiment failed or timed out: $experiment_name"
        echo "EXPERIMENT_FAILED_OR_TIMEOUT" >> "$log_file"
    fi
    
    echo "================================================"
}

# Function to run quick validation tests
run_quick_tests() {
    print_status "Running quick validation tests..."
    
    # Quick test with 5 epochs each
    run_experiment "configs/cifar100/sdd_lskd/quick_test.yaml" "[1]" "global_lskd" "cifar100" "quick"
    run_experiment "configs/cifar100/sdd_lskd/quick_test.yaml" "[1,2]" "twoscale_sdd_lskd" "cifar100" "quick"
    run_experiment "configs/cifar100/sdd_lskd/quick_test.yaml" "[1,2,4]" "threescale_sdd_lskd" "cifar100" "quick"
    
    print_success "Quick validation tests completed"
}

# Function to run full CIFAR-100 experiments
run_cifar100_full() {
    print_status "Running full CIFAR-100 experiments..."
    
    # ResNet32x4 -> ResNet8x4 (Homogeneous)
    print_status "ResNet32x4 -> ResNet8x4 experiments"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1]" "res32x4_res8x4_global" "cifar100" "full"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1,2]" "res32x4_res8x4_twoscale" "cifar100" "full"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1,2,4]" "res32x4_res8x4_threescale" "cifar100" "full"
    
    # ResNet32x4 -> MobileNetV2 (Heterogeneous)
    print_status "ResNet32x4 -> MobileNetV2 experiments"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_mv2.yaml" "[1]" "res32x4_mv2_global" "cifar100" "full"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_mv2.yaml" "[1,2]" "res32x4_mv2_twoscale" "cifar100" "full"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_mv2.yaml" "[1,2,4]" "res32x4_mv2_threescale" "cifar100" "full"
    
    # ResNet32x4 -> ShuffleNetV1 (Heterogeneous)
    print_status "ResNet32x4 -> ShuffleNetV1 experiments"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1]" "res32x4_shuv1_global" "cifar100" "full"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1,2]" "res32x4_shuv1_twoscale" "cifar100" "full"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_shuv1.yaml" "[1,2,4]" "res32x4_shuv1_threescale" "cifar100" "full"
    
    print_success "CIFAR-100 full experiments completed"
}

# Function to run ablation studies
run_ablation_studies() {
    print_status "Running ablation studies..."
    
    # Ablation on multi-scale settings
    print_status "Multi-scale ablation study"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1]" "ablation_single_scale" "cifar100" "ablation"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[2]" "ablation_scale2_only" "cifar100" "ablation"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[4]" "ablation_scale4_only" "cifar100" "ablation"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[1,4]" "ablation_scale1_4" "cifar100" "ablation"
    run_experiment "configs/cifar100/sdd_lskd/res32x4_res8x4.yaml" "[2,4]" "ablation_scale2_4" "cifar100" "ablation"
    
    print_success "Ablation studies completed"
}

# Function to run baseline comparisons
run_baseline_comparisons() {
    print_status "Running baseline comparisons..."
    
    # Traditional KD methods
    print_status "Traditional knowledge distillation baselines"
    run_experiment "configs/cifar100/kd.yaml" "" "baseline_kd" "cifar100" "comparison"
    run_experiment "configs/cifar100/dkd/res32x4_res8x4.yaml" "" "baseline_dkd" "cifar100" "comparison"
    
    # Individual SDD and LSKD
    print_status "Individual method comparisons"
    run_experiment "configs/cifar100/sdd_kd/res32x4_res8x4.yaml" "[1,2,4]" "baseline_sdd_kd" "cifar100" "comparison"
    
    print_success "Baseline comparisons completed"
}

# Function to generate experiment summary
generate_summary() {
    print_status "Generating experiment summary..."
    
    local summary_file="$BASE_LOG_DIR/summaries/experiment_summary_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$summary_file" << EOF
SDD-LSKD Experimental Results Summary
=====================================
Generated: $(date)
Experiment Set: $EXPERIMENT_SET

Experiment Structure:
- Quick Tests: Validation with 5 epochs
- Full Experiments: Complete 240 epoch training
- Ablation Studies: Multi-scale setting analysis
- Baseline Comparisons: Against traditional KD methods

Results Analysis:
EOF
    
    # Extract results from log files
    print_status "Extracting results from log files..."
    
    for log_dir in "$BASE_LOG_DIR"/sdd_lskd_*; do
        if [ -d "$log_dir" ]; then
            echo "" >> "$summary_file"
            echo "=== $(basename "$log_dir") ===" >> "$summary_file"
            
            for log_file in "$log_dir"/*.log; do
                if [ -f "$log_file" ]; then
                    local exp_name=$(basename "$log_file" .log)
                    local final_acc=$(tail -20 "$log_file" | grep -oE "Test Acc: [0-9]+\.[0-9]+" | tail -1 | grep -oE "[0-9]+\.[0-9]+")
                    
                    if [ -n "$final_acc" ]; then
                        echo "$exp_name: $final_acc%" >> "$summary_file"
                    else
                        echo "$exp_name: No final accuracy found" >> "$summary_file"
                    fi
                fi
            done
        fi
    done
    
    print_success "Summary generated: $summary_file"
}

# Function to display usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --set <quick|full|ablation|comparison|all>  Set experiment type (default: full)"
    echo "  --gpu <id>                                  GPU ID to use (default: 0)"
    echo "  --timeout <seconds>                         Timeout per experiment (default: 7200)"
    echo "  --help                                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --set quick              # Run quick validation tests"
    echo "  $0 --set full --gpu 1       # Run full experiments on GPU 1"
    echo "  $0 --set all                # Run all experiment types"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --set)
            EXPERIMENT_SET="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT_DURATION="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_status "Starting SDD-LSKD experimental suite"
    print_status "Experiment set: $EXPERIMENT_SET"
    print_status "GPU ID: $GPU_ID"
    print_status "Timeout per experiment: $TIMEOUT_DURATION seconds"
    echo ""
    
    # Setup
    setup_environment
    setup_logging
    
    # Run experiments based on selected set
    case $EXPERIMENT_SET in
        quick)
            run_quick_tests
            ;;
        full)
            run_cifar100_full
            ;;
        ablation)
            run_ablation_studies
            ;;
        comparison)
            run_baseline_comparisons
            ;;
        all)
            run_quick_tests
            run_cifar100_full
            run_ablation_studies
            run_baseline_comparisons
            ;;
        *)
            print_error "Invalid experiment set: $EXPERIMENT_SET"
            show_usage
            exit 1
            ;;
    esac
    
    # Generate summary
    generate_summary
    
    print_success "All experiments completed successfully!"
    print_status "Check log files in $BASE_LOG_DIR/ for detailed results"
}

# Trap to handle interruption
trap 'print_warning "Experiment interrupted by user"; exit 130' INT

# Run main function
main "$@"
