#!/bin/bash

# ============================================================================
# Group Experiment - Training & Evaluation Pipeline
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAMES=(
    "HuggingFaceTB/SmolLM2-1.7B"
    # "TinyLlama/TinyLlama_v1.1"
    # "microsoft/phi-1_5"
)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="${TRAIN_MODE:-auto}"

if [[ "$MODE" == "cluster" ]] || [[ -n "${SLURM_JOB_ID:-}" ]] || [[ -n "${APPTAINER_NAME:-${SINGULARITY_NAME:-}}" ]]; then
    BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:?BASE_OUTPUT_DIR not set}"
else
    REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$REPO_ROOT/train_results_group_exp}"
fi
export BASE_OUTPUT_DIR
echo "BASE_OUTPUT_DIR=$BASE_OUTPUT_DIR"

# --- Iteration Parameters ---
TRAINING_MODES=("qalora" "qalora_svd_error" "sa_svd") 
LORA_RANKS=(4 8 16 32 64)
BITS_LIST=(2)
CALIBRATION_DATASETS=("c4")
# Hier iterieren wir über die Group Sizes
QALORA_GROUP_SIZES=(16 32)

# --- Training Configuration ---
DATA_PATH="yahma/alpaca-cleaned"
DATASET_SPLIT="train[:10000]"
DATASET_VAL_SPLIT="train[30000:31000]"
NUM_TRAIN_EPOCHS=1
PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-4
MAX_LENGTH=2048
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=10
SAVE_STEPS=5000
BF16="True"

# ============================================================================
# Helper Functions
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ============================================================================
# Main Execution
# ============================================================================
main() {
    log_info "Starting Group Experiment Pipeline"
    mkdir -p "$BASE_OUTPUT_DIR"
    cd "$SCRIPT_DIR" || exit 1

    for MODEL_NAME_OR_PATH in "${MODEL_NAMES[@]}"; do
        MODEL_SHORT_NAME="${MODEL_NAME_OR_PATH##*/}"
        
        # WICHTIG: Projektname generisch halten (nicht hardcodiert auf 16),
        # damit du später im Dashboard nach group_size filtern kannst.
        export WANDB_PROJECT="qalora-finetuning-${MODEL_SHORT_NAME}-group-variation"

        log_info "========================================="
        log_info "Processing Model: ${MODEL_SHORT_NAME}"
        log_info "========================================="

        for mode in "${TRAINING_MODES[@]}"; do
            for rank in "${LORA_RANKS[@]}"; do
                for bits in "${BITS_LIST[@]}"; do
                    for dataset in "${CALIBRATION_DATASETS[@]}"; do
                        
                        GROUP_SIZES_TO_RUN=("${QALORA_GROUP_SIZES[@]}")

                        for group_size in "${GROUP_SIZES_TO_RUN[@]}"; do
                            log_info "===== Running: MODEL=${MODEL_SHORT_NAME} MODE=${mode} RANK=${rank} BITS=${bits} DATASET=${dataset} GROUP=${group_size} ====="

                            EXPERIMENT_NAME="${MODEL_SHORT_NAME}_${mode}_r${rank}_b${bits}_d${dataset}_group${group_size}"
                            TRAIN_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${EXPERIMENT_NAME}"
                            ADAPTER_DIR="${TRAIN_OUTPUT_DIR}/adapter"

                            # --- W&B CONFIG INJECTION ---
                            # 1. WANDB_NAME: Damit der Run in der Liste genau so heißt wie das Experiment
                            export WANDB_NAME="$EXPERIMENT_NAME"
                            
                            # 2. WANDB_CONFIG_...: Das zwingt W&B, diesen Parameter in die "Config"-Spalte zu schreiben,
                            # selbst wenn das Python-Skript es vergessen würde.
                            export WANDB_CONFIG_qalora_group_size="$group_size"
                            export WANDB_CONFIG_training_mode="$mode"
                            export WANDB_CONFIG_lora_rank="$rank"

                            if [ -d "$ADAPTER_DIR" ]; then
                                log_warning "Adapter exists at $ADAPTER_DIR. Skipping training."
                            else
                                log_info "Starting training for $EXPERIMENT_NAME"
                                
                                # Wir übergeben group_size natürlich auch weiterhin an das Python Skript
                                python ultimate_train_collection.py \
                                    --model_name_or_path="$MODEL_NAME_OR_PATH" \
                                    --training_mode="$mode" \
                                    --output_dir="$TRAIN_OUTPUT_DIR" \
                                    --data_path="$DATA_PATH" \
                                    --dataset_split="$DATASET_SPLIT" \
                                    --dataset_split_validation="$DATASET_VAL_SPLIT" \
                                    --dataset_field "instruction" "output" \
                                    --lora_r="$rank" \
                                    --qalora_group_size="$group_size" \
                                    --bits="$bits" \
                                    --calibration_dataset="$dataset" \
                                    --num_train_epochs="$NUM_TRAIN_EPOCHS" \
                                    --per_device_train_batch_size="$PER_DEVICE_TRAIN_BATCH_SIZE" \
                                    --gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS" \
                                    --learning_rate="$LEARNING_RATE" \
                                    --lr_scheduler_type="$LR_SCHEDULER_TYPE" \
                                    --warmup_ratio="$WARMUP_RATIO" \
                                    --bf16="$BF16" \
                                    --logging_steps="$LOGGING_STEPS" \
                                    --save_steps="$SAVE_STEPS" \
                                    --model_max_length="$MAX_LENGTH" \
                                    --eval_steps=100 \
                                    --report_to="wandb" 

                                if [ $? -ne 0 ]; then
                                    log_error "Training failed for $EXPERIMENT_NAME. Skipping."
                                    continue
                                fi
                                log_success "Training & Evaluation completed for $EXPERIMENT_NAME"
                            fi
                            
                            # Kurze Pause für W&B Sync Sicherheit
                            sleep 5
                        done
                    done
                done
            done
        done
    done

    log_success "🎉 Group Experiment Pipeline Finished! 🎉"
}

main "$@"