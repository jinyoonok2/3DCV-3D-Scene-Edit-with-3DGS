#!/bin/bash

# InstructGS Example Scripts
# This script demonstrates various usage patterns for InstructGS

set -e  # Exit on any error

# Configuration
DATA_DIR="datasets"
OUTPUT_DIR="outputs/instruct_gs_examples"
VIEWER_PORT=7007

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE} $1 ${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if dataset exists
check_dataset() {
    local dataset_path="$1"
    if [ ! -d "$dataset_path" ]; then
        print_error "Dataset not found: $dataset_path"
        print_info "Please ensure your dataset is processed with nerfstudio:"
        print_info "  ns-process-data images --data $dataset_path --output-dir $dataset_path"
        return 1
    fi
    return 0
}

# Example 1: Basic painting transformation
example_painting() {
    print_header "Example 1: Turn bicycle into a painting"
    
    local dataset="$DATA_DIR/bicycle"
    if ! check_dataset "$dataset"; then
        return 1
    fi
    
    print_info "Running: Turn bicycle into a painting"
    print_info "This will take approximately 2-4 hours"
    
    python instruct_gs/train_instruct_gs.py \
        --data "$dataset" \
        --edit-prompt "Turn it into a beautiful oil painting" \
        --cycle-steps 2500 \
        --max-num-iterations 30000 \
        --output-dir "$OUTPUT_DIR" \
        --experiment-name "bicycle_painting" \
        --viewer-port $VIEWER_PORT
}

# Example 2: Winter transformation
example_winter() {
    print_header "Example 2: Make garden winter scene"
    
    local dataset="$DATA_DIR/garden"
    if ! check_dataset "$dataset"; then
        return 1
    fi
    
    print_info "Running: Make garden into winter scene"
    print_info "Using faster cycle for seasonal changes"
    
    python instruct_gs/train_instruct_gs.py \
        --data "$dataset" \
        --edit-prompt "Make it winter with snow covering everything" \
        --cycle-steps 2000 \
        --max-num-iterations 25000 \
        --ip2p-guidance-scale 8.0 \
        --output-dir "$OUTPUT_DIR" \
        --experiment-name "garden_winter" \
        --viewer-port $((VIEWER_PORT + 1))
}

# Example 3: Cartoon style
example_cartoon() {
    print_header "Example 3: Turn room into cartoon style"
    
    local dataset="$DATA_DIR/room"
    if ! check_dataset "$dataset"; then
        return 1
    fi
    
    print_info "Running: Turn room into cartoon style"
    print_info "Using longer cycles for style transfer"
    
    python instruct_gs/train_instruct_gs.py \
        --data "$dataset" \
        --edit-prompt "Turn it into a cartoon style with vibrant colors" \
        --cycle-steps 3000 \
        --max-num-iterations 35000 \
        --ip2p-guidance-scale 7.0 \
        --ip2p-image-guidance-scale 2.0 \
        --output-dir "$OUTPUT_DIR" \
        --experiment-name "room_cartoon" \
        --viewer-port $((VIEWER_PORT + 2))
}

# Example 4: Using pre-trained model
example_pretrained() {
    print_header "Example 4: Edit using pre-trained SplatfactoModel"
    
    local dataset="$DATA_DIR/bicycle"
    local pretrained_dir="outputs/bicycle/splatfacto"
    
    if ! check_dataset "$dataset"; then
        return 1
    fi
    
    if [ ! -d "$pretrained_dir" ]; then
        print_warning "Pre-trained model not found: $pretrained_dir"
        print_info "First train a base SplatfactoModel:"
        print_info "  ns-train splatfacto --data $dataset"
        print_info "Then use the checkpoint directory as --load-dir"
        return 1
    fi
    
    # Find the latest checkpoint
    local latest_checkpoint=$(find "$pretrained_dir" -type d -name "20*" | sort | tail -1)
    
    if [ -z "$latest_checkpoint" ]; then
        print_error "No checkpoint found in $pretrained_dir"
        return 1
    fi
    
    print_info "Using pre-trained model: $latest_checkpoint"
    print_info "Running: Set bicycle on fire"
    
    python instruct_gs/train_instruct_gs.py \
        --data "$dataset" \
        --load-dir "$latest_checkpoint" \
        --edit-prompt "Set it on fire with dramatic flames" \
        --cycle-steps 2000 \
        --max-num-iterations 20000 \
        --ip2p-guidance-scale 9.0 \
        --output-dir "$OUTPUT_DIR" \
        --experiment-name "bicycle_fire" \
        --viewer-port $((VIEWER_PORT + 3))
}

# Example 5: Quick test with small dataset
example_quick_test() {
    print_header "Example 5: Quick test (small iterations)"
    
    local dataset="$DATA_DIR/bicycle"
    if ! check_dataset "$dataset"; then
        return 1
    fi
    
    print_info "Running: Quick test with minimal iterations"
    print_info "This will complete in ~30 minutes for testing"
    
    python instruct_gs/train_instruct_gs.py \
        --data "$dataset" \
        --edit-prompt "Turn it into a sketch drawing" \
        --cycle-steps 1000 \
        --max-num-iterations 5000 \
        --steps-per-save 500 \
        --steps-per-eval 500 \
        --output-dir "$OUTPUT_DIR" \
        --experiment-name "bicycle_sketch_test" \
        --viewer-port $((VIEWER_PORT + 4))
}

# Example 6: Multiple GPU setup
example_multi_gpu() {
    print_header "Example 6: Multi-GPU setup"
    
    local dataset="$DATA_DIR/garden"
    if ! check_dataset "$dataset"; then
        return 1
    fi
    
    # Check if multiple GPUs are available
    if ! command -v nvidia-smi &> /dev/null; then
        print_warning "nvidia-smi not available, skipping multi-GPU example"
        return 1
    fi
    
    local gpu_count=$(nvidia-smi --list-gpus | wc -l)
    if [ "$gpu_count" -lt 2 ]; then
        print_warning "Only $gpu_count GPU(s) available, skipping multi-GPU example"
        return 1
    fi
    
    print_info "Running: Multi-GPU setup with IP2P on separate GPU"
    print_info "Main model on cuda:0, InstructPix2Pix on cuda:1"
    
    python instruct_gs/train_instruct_gs.py \
        --data "$dataset" \
        --edit-prompt "Transform into a magical fairy tale scene" \
        --cycle-steps 2500 \
        --ip2p-device "cuda:1" \
        --max-num-iterations 30000 \
        --output-dir "$OUTPUT_DIR" \
        --experiment-name "garden_fairy_tale" \
        --viewer-port $((VIEWER_PORT + 5))
}

# Function to show usage
show_usage() {
    print_header "InstructGS Example Scripts"
    echo ""
    echo "Usage: $0 [EXAMPLE_NUMBER]"
    echo ""
    echo "Available examples:"
    echo "  1. painting     - Turn bicycle into a painting"
    echo "  2. winter       - Make garden into winter scene"
    echo "  3. cartoon      - Turn room into cartoon style"
    echo "  4. pretrained   - Edit using pre-trained model"
    echo "  5. quick        - Quick test with minimal iterations"
    echo "  6. multi-gpu    - Multi-GPU setup example"
    echo "  all            - Run all examples (sequential)"
    echo ""
    echo "Examples:"
    echo "  $0 1            # Run painting example"
    echo "  $0 cartoon      # Run cartoon example"
    echo "  $0 all          # Run all examples"
    echo ""
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    case "$1" in
        1|painting)
            example_painting
            ;;
        2|winter)
            example_winter
            ;;
        3|cartoon)
            example_cartoon
            ;;
        4|pretrained)
            example_pretrained
            ;;
        5|quick)
            example_quick_test
            ;;
        6|multi-gpu)
            example_multi_gpu
            ;;
        all)
            print_header "Running all InstructGS examples"
            print_warning "This will take many hours to complete!"
            read -p "Continue? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                example_painting
                example_winter  
                example_cartoon
                example_pretrained
                example_quick_test
                example_multi_gpu
                print_header "All examples completed!"
            else
                print_info "Cancelled by user"
            fi
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown example: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"