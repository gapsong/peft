#!/bin/bash

# ============================================================================
# PiSSA Residual Quantization - Training & Evaluation Pipeline
# ============================================================================
# This script trains PiSSA models with different ranks and evaluates them
# across multiple quantization configurations.

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="/home/nudel/Documents/peft/examples/qalora_finetuning"
BASE_OUTPUT_DIR="/home/nudel/Documents/peft/train_results_debugger"
LORA_RANKS=(1 2 4 8 16 32 64 512)
CUDA_DEVICE="0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to fix adapter config
fix_adapter_config() {
    local adapter_path="$1"
    local config_file="$adapter_path/adapter_config.json"
    
    if [[ -f "$config_file" ]]; then
        log_info "Fixing adapter config: $config_file"
        
        # Create backup
        cp "$config_file" "$config_file.backup"
        
        # Use jq to set init_lora_weights to false
        if command -v jq &> /dev/null; then
            jq '.init_lora_weights = false' "$config_file" > "$config_file.tmp" && mv "$config_file.tmp" "$config_file"
            log_success "Fixed init_lora_weights in $config_file"
        else
            # Fallback: use sed
            log_warning "jq not found, using sed fallback"
            sed -i 's/"init_lora_weights": "daniel"/"init_lora_weights": false/g' "$config_file"
            sed -i 's/"init_lora_weights": true/"init_lora_weights": false/g' "$config_file"
            log_success "Fixed init_lora_weights using sed"
        fi
    else
        log_error "Adapter config file not found: $config_file"
    fi
}

# Function to find the actual residual models directory
find_actual_residual_dir() {
    local base_dir="$1"
    local rank="$2"
    
    # Check different possible paths due to nesting
    local possible_paths=(
        "$base_dir/quantized_residuals_r${rank}"
        "$base_dir/quantized_residuals_r${rank}/quantized_residuals_r${rank}"
        "$base_dir/quantized_residuals/quantized_residuals_r${rank}"
    )
    
    for path in "${possible_paths[@]}"; do
        if [[ -d "$path" ]]; then
            # Check if this directory contains actual model files
            local has_models=false
            if ls "$path"/w_res_* &>/dev/null || ls "$path"/daniel_adapter_* &>/dev/null; then
                has_models=true
            fi
            
            if $has_models; then
                echo "$path"
                return 0
            fi
        fi
    done
    
    log_error "Could not find actual residual models directory for rank $rank"
    log_error "Searched in: ${possible_paths[*]}"
    return 1
}

# Function to train model with specific rank
train_model() {
    local rank=$1
    local output_dir="$BASE_OUTPUT_DIR/quantized_residuals_r${rank}"
    
    log_info "Starting training for rank $rank..."
    log_info "Output directory: $output_dir"
    
    cd /home/nudel/Documents/peft
    
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    export PYTHONPATH="/home/nudel/Documents/peft:${PYTHONPATH}"
    
    python "$SCRIPT_DIR/corda_finetuning.py" \
        --model_name_or_path="HuggingFaceTB/SmolLM2-1.7B" \
        --output_dir="$output_dir" \
        --data_path="yahma/alpaca-cleaned" \
        --dataset_split="train[:10000]" \
        --dataset_field "instruction" "output" \
        --bits=2 \
        --lora_r=$rank \
        --num_train_epochs=2 \
        --per_device_train_batch_size=4 \
        --gradient_accumulation_steps=1 \
        --learning_rate=1e-4 \
        --lr_scheduler_type="cosine" \
        --warmup_ratio=0.03 \
        --bf16=True \
        --logging_steps=10 \
        --save_steps=5000 \
        --eval_steps=500 \
        --training_mode="pissa_rank_analysis" \
        --dataloader_pin_memory=False \
        --remove_unused_columns=False \
        --report_to="none"
    
    if [[ $? -eq 0 ]]; then
        log_success "Training completed for rank $rank"
        
        # Find the actual adapter path (might be nested)
        local actual_residual_dir
        actual_residual_dir=$(find_actual_residual_dir "$BASE_OUTPUT_DIR" "$rank")
        
        if [[ $? -eq 0 ]]; then
            local adapter_path="$actual_residual_dir/daniel_adapter_r${rank}_HuggingFaceTB_SmolLM2-1.7B"
            if [[ -d "$adapter_path" ]]; then
                fix_adapter_config "$adapter_path"
            else
                log_warning "Adapter path not found: $adapter_path"
                # Try to find it with find command
                local found_adapter=$(find "$output_dir" -name "daniel_adapter_r${rank}_*" -type d 2>/dev/null | head -1)
                if [[ -n "$found_adapter" ]]; then
                    log_info "Found adapter at: $found_adapter"
                    fix_adapter_config "$found_adapter"
                fi
            fi
        fi
    else
        log_error "Training failed for rank $rank"
        return 1
    fi
}

# Function to evaluate model with specific rank
evaluate_model() {
    local rank=$1
    local eval_output_dir="./residual_evaluation_results_r${rank}"
    
    log_info "Starting evaluation for rank $rank..."
    
    # Find the actual residual models directory
    local residual_dir
    residual_dir=$(find_actual_residual_dir "$BASE_OUTPUT_DIR" "$rank")
    
    if [[ $? -ne 0 ]]; then
        log_error "Could not find residual models directory for rank $rank"
        return 1
    fi
    
    log_info "Found residual models dir: $residual_dir"
    log_info "Evaluation output dir: $eval_output_dir"
    
    # List what's in the directory for debugging
    log_info "Contents of residual directory:"
    ls -la "$residual_dir" || true
    
    cd /home/nudel/Documents/peft
    
    export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
    export PYTHONPATH="/home/nudel/Documents/peft:${PYTHONPATH}"
    
    python "$SCRIPT_DIR/evaluate_residual_models.py" \
        --residual_models_dir="$residual_dir" \
        --output_dir="$eval_output_dir" \
        --tasks="hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq,openbookqa,wikitext" \
        --num_fewshot=5 \
        --limit=100 \
        --save_results \
        --generate_latex
    
    if [[ $? -eq 0 ]]; then
        log_success "Evaluation completed for rank $rank"
        log_info "Results saved to: $eval_output_dir"
    else
        log_error "Evaluation failed for rank $rank"
        return 1
    fi
}

# Function to create combined results summary
create_combined_summary() {
    log_info "Creating combined results summary..."
    
    local summary_dir="./combined_residual_analysis_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$summary_dir"
    
    # Copy all individual results
    for rank in "${LORA_RANKS[@]}"; do
        local eval_dir="./residual_evaluation_results_r${rank}"
        if [[ -d "$eval_dir" ]]; then
            cp -r "$eval_dir" "$summary_dir/rank_${rank}_results"
            log_info "Copied results for rank $rank"
        fi
    done
    
    # Create a master summary file
    cat > "$summary_dir/README.md" << EOF
# PiSSA Residual Quantization Analysis Results

Generated on: $(date)

## Experiment Configuration
- Model: HuggingFaceTB/SmolLM2-1.7B
- Training Mode: PiSSA Rank Analysis
- LoRA Ranks Tested: ${LORA_RANKS[*]}
- Quantization Bits: 2, 3, 4 (with various group sizes)
- Evaluation Tasks: hellaswag, piqa, winogrande, arc_easy, arc_challenge, boolq, openbookqa, wikitext
- Few-shot Examples: 5
- Sample Limit: 100

## Directory Structure
EOF
    
    for rank in "${LORA_RANKS[@]}"; do
        echo "- \`rank_${rank}_results/\`: Results for LoRA rank $rank" >> "$summary_dir/README.md"
    done
    
    log_success "Combined summary created in: $summary_dir"
}

# Main execution
main() {
    log_info "Starting PiSSA Residual Quantization Pipeline"
    log_info "LoRA ranks to process: ${LORA_RANKS[*]}"
    log_info "Base output directory: $BASE_OUTPUT_DIR"
    
    # Check if required directories exist
    if [[ ! -f "$SCRIPT_DIR/corda_finetuning.py" ]]; then
        log_error "Training script not found: $SCRIPT_DIR/corda_finetuning.py"
        exit 1
    fi
    
    if [[ ! -f "$SCRIPT_DIR/evaluate_residual_models.py" ]]; then
        log_error "Evaluation script not found: $SCRIPT_DIR/evaluate_residual_models.py"
        exit 1
    fi
    
    # Create base output directory
    mkdir -p "$BASE_OUTPUT_DIR"
    
    # Find existing models (since training is commented out)
    log_info "===== FINDING EXISTING MODELS ====="
    successful_training=()
    
    for rank in "${LORA_RANKS[@]}"; do
        local residual_dir
        residual_dir=$(find_actual_residual_dir "$BASE_OUTPUT_DIR" "$rank")
        
        if [[ $? -eq 0 ]]; then
            log_success "Found existing models for rank $rank at: $residual_dir"
            successful_training+=($rank)
            
            # Fix adapter config if needed
            local adapter_path="$residual_dir/daniel_adapter_r${rank}_HuggingFaceTB_SmolLM2-1.7B"
            if [[ -d "$adapter_path" ]]; then
                fix_adapter_config "$adapter_path"
            else
                # Try to find it with find command
                local found_adapter=$(find "$residual_dir" -name "daniel_adapter_r${rank}_*" -type d 2>/dev/null | head -1)
                if [[ -n "$found_adapter" ]]; then
                    log_info "Found adapter at: $found_adapter"
                    fix_adapter_config "$found_adapter"
                fi
            fi
        else
            log_warning "No existing models found for rank $rank"
        fi
    done
    
    log_info "Found existing models for ranks: ${successful_training[*]}"
    
    # Phase 2: Evaluation
    log_info "===== PHASE 2: EVALUATING MODELS ====="
    failed_evaluation=()
    successful_evaluation=()
    
    for rank in "${successful_training[@]}"; do
        log_info "Evaluating rank $rank (${#successful_evaluation[@]}/${#successful_training[@]} completed)"
        
        if evaluate_model $rank; then
            successful_evaluation+=($rank)
            log_success "✅ Evaluation successful for rank $rank"
        else
            failed_evaluation+=($rank)
            log_error "❌ Evaluation failed for rank $rank"
        fi
        
        # Small delay between evaluation runs
        sleep 5
    done
    
    # Evaluation summary
    log_info "===== EVALUATION SUMMARY ====="
    log_success "Successful evaluation: ${successful_evaluation[*]}"
    if [[ ${#failed_evaluation[@]} -gt 0 ]]; then
        log_error "Failed evaluation: ${failed_evaluation[*]}"
    fi
    
    # Phase 3: Create combined summary
    if [[ ${#successful_evaluation[@]} -gt 0 ]]; then
        log_info "===== PHASE 3: CREATING COMBINED SUMMARY ====="
        create_combined_summary
    fi
    
    # Final summary
    log_info "===== PIPELINE COMPLETED ====="
    log_success "Successfully processed ranks: ${successful_evaluation[*]}"
    log_info "Total models found: ${#successful_training[@]}/${#LORA_RANKS[@]}"
    log_info "Total models evaluated: ${#successful_evaluation[@]}/${#successful_training[@]}"
    
    if [[ ${#successful_evaluation[@]} -eq ${#LORA_RANKS[@]} ]]; then
        log_success "🎉 All models processed successfully!"
    else
        log_warning "⚠️  Some models failed. Check logs above for details."
    fi
}

# Script execution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Check for required tools
    if ! command -v python &> /dev/null; then
        log_error "Python not found in PATH"
        exit 1
    fi
    
    # Run main function
    main "$@"
fi