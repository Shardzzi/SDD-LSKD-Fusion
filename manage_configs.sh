#!/bin/bash

# SDD-LSKD Configuration and Experiment Management
# This script manages the complete set of SDD-LSKD configurations

# List all available configurations
echo "Available SDD-LSKD Configurations:"
echo "=================================="

# Quick test configurations
echo ""
echo "Quick Test Configurations (5 epochs):"
echo "- configs/cifar100/sdd_lskd/quick_test.yaml"
echo "- configs/cifar100/sdd_lskd/cpu_quick_test.yaml"

# Full training configurations
echo ""
echo "Full Training Configurations (240 epochs):"
echo "ResNet architectures:"
echo "- configs/cifar100/sdd_lskd/res32x4_res8x4.yaml    # ResNet32x4 -> ResNet8x4"
echo "- configs/cifar100/sdd_lskd/res110_res32.yaml      # ResNet110 -> ResNet32"
echo "- configs/cifar100/sdd_lskd/res56_res20.yaml       # ResNet56 -> ResNet20"

echo ""
echo "Heterogeneous architectures:"
echo "- configs/cifar100/sdd_lskd/res32x4_mv2.yaml       # ResNet32x4 -> MobileNetV2"
echo "- configs/cifar100/sdd_lskd/res32x4_shuv1.yaml     # ResNet32x4 -> ShuffleNetV1"
echo "- configs/cifar100/sdd_lskd/res32x4_shuv2.yaml     # ResNet32x4 -> ShuffleNetV2"
echo "- configs/cifar100/sdd_lskd/res50_mv2.yaml         # ResNet50 -> MobileNetV2"
echo "- configs/cifar100/sdd_lskd/resnet32_mobinetv2.yaml # ResNet32 -> MobileNetV2"

echo ""
echo "VGG architectures:"
echo "- configs/cifar100/sdd_lskd/vgg13_mv2.yaml         # VGG13 -> MobileNetV2"
echo "- configs/cifar100/sdd_lskd/vgg13_shuffv1.yaml     # VGG13 -> ShuffleNetV1"
echo "- configs/cifar100/sdd_lskd/vgg13_vgg8.yaml        # VGG13 -> VGG8"
echo "- configs/cifar100/sdd_lskd/vgg13_wrn_16_2.yaml    # VGG13 -> WRN_16_2"
echo "- configs/cifar100/sdd_lskd/resnet32x4_vgg8.yaml   # ResNet32x4 -> VGG8"

echo ""
echo "Wide ResNet architectures:"
echo "- configs/cifar100/sdd_lskd/wrn40_2_shuv1.yaml     # WRN_40_2 -> ShuffleNetV1"
echo "- configs/cifar100/sdd_lskd/wrn40_2_wrn_16_2.yaml  # WRN_40_2 -> WRN_16_2"
echo "- configs/cifar100/sdd_lskd/wrn40_2_wrn_40_1.yaml  # WRN_40_2 -> WRN_40_1"

echo ""
echo "Multi-scale Settings:"
echo "====================="
echo "[1]       - Global distillation only (LSKD standardization)"
echo "[1,2]     - Two-scale: Global + 2x2 blocks (SDD + LSKD fusion)"
echo "[1,2,4]   - Three-scale: Global + 2x2 + 4x4 blocks (Complete SDD + LSKD fusion)"
echo "[2]       - 2x2 blocks only (for ablation studies)"
echo "[4]       - 4x4 blocks only (for ablation studies)"
echo "[1,4]     - Global + 4x4 blocks (skip 2x2 scale)"
echo "[2,4]     - 2x2 + 4x4 blocks (skip global scale)"

echo ""
echo "Usage Examples:"
echo "==============="
echo "# Quick validation test"
echo "./quick_validation.sh"
echo ""
echo "# Single experiment"
echo "./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml \"[1,2,4]\" \"test_fusion\""
echo ""
echo "# Complete experimental suite"
echo "./run_complete_experiments.sh --set full --gpu 0"
echo ""
echo "# Ablation study"
echo "./run_complete_experiments.sh --set ablation"

# Function to validate configuration file
validate_config() {
    local config_file=$1
    
    if [ ! -f "$config_file" ]; then
        echo "Error: Configuration file not found: $config_file"
        return 1
    fi
    
    # Check required fields
    if ! grep -q "DISTILLER:" "$config_file"; then
        echo "Error: DISTILLER section missing in $config_file"
        return 1
    fi
    
    if ! grep -q "TYPE: \"SDD_LSKD\"" "$config_file"; then
        echo "Error: Not an SDD_LSKD configuration: $config_file"
        return 1
    fi
    
    echo "Valid SDD-LSKD configuration: $config_file"
    return 0
}

# Function to list all configurations
list_configs() {
    echo ""
    echo "Validating all SDD-LSKD configurations..."
    echo "========================================"
    
    local config_dir="configs/cifar100/sdd_lskd"
    local valid_count=0
    local total_count=0
    
    for config_file in "$config_dir"/*.yaml; do
        if [ -f "$config_file" ]; then
            total_count=$((total_count + 1))
            if validate_config "$config_file"; then
                valid_count=$((valid_count + 1))
            fi
        fi
    done
    
    echo ""
    echo "Summary: $valid_count/$total_count configurations are valid"
}

# Function to create experiment plan
create_experiment_plan() {
    local plan_file="experiment_plan_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > "$plan_file" << EOF
SDD-LSKD Experiment Plan
========================
Generated: $(date)

Recommended Experiment Sequence:

1. Quick Validation (15-30 minutes)
   Purpose: Verify environment and method functionality
   Command: ./quick_validation.sh
   
   Tests:
   - Global LSKD (M=[1])
   - Two-scale SDD-LSKD (M=[1,2]) 
   - Three-scale SDD-LSKD (M=[1,2,4])

2. Baseline Experiments (8-12 hours each)
   Purpose: Establish core performance metrics
   
   ResNet32x4 -> ResNet8x4:
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml "[1]" "baseline_global"
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml "[1,2]" "baseline_twoscale"
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml "[1,2,4]" "baseline_threescale"
   
   ResNet32x4 -> MobileNetV2:
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_mv2.yaml "[1]" "hetero_global"
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_mv2.yaml "[1,2]" "hetero_twoscale"
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_mv2.yaml "[1,2,4]" "hetero_threescale"

3. Ablation Studies (4-6 hours each)
   Purpose: Understand contribution of each component
   
   Scale ablation:
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml "[2]" "ablation_scale2_only"
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml "[4]" "ablation_scale4_only"
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml "[1,4]" "ablation_skip_scale2"
   ./run_single_experiment.sh configs/cifar100/sdd_lskd/res32x4_res8x4.yaml "[2,4]" "ablation_skip_global"

4. Comprehensive Evaluation (24-48 hours)
   Purpose: Evaluate across multiple architectures
   Command: ./run_complete_experiments.sh --set full

5. Comparison with Baselines
   Purpose: Compare against state-of-the-art methods
   Command: ./run_complete_experiments.sh --set comparison

Expected Results:
- M=[1]: Baseline LSKD performance
- M=[1,2]: 1-2% improvement over M=[1]
- M=[1,2,4]: 2-3% improvement over M=[1], best overall performance
- Heterogeneous architectures: Greater benefits from multi-scale fusion
- VGG architectures: Significant improvements due to architectural differences

Total Estimated Time: 3-7 days for complete evaluation
EOF
    
    echo "Experiment plan created: $plan_file"
}

# Main execution
case "${1:-list}" in
    "list")
        list_configs
        ;;
    "plan")
        create_experiment_plan
        ;;
    "validate")
        if [ -n "$2" ]; then
            validate_config "$2"
        else
            echo "Usage: $0 validate <config_file>"
        fi
        ;;
    *)
        echo "Usage: $0 [list|plan|validate]"
        echo "  list     - List and validate all configurations"
        echo "  plan     - Create detailed experiment plan"
        echo "  validate - Validate specific configuration file"
        ;;
esac
