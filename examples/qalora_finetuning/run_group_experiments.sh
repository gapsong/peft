#!/bin/bash
# filepath: /home/gap/Documents/peft/examples/qalora_finetuning/run_group_experiments.sh

# Configuration
BASE_MODEL="HuggingFaceTB/SmolLM2-1.7B"
DATASET="yahma/alpaca-cleaned"
DATASET_SPLIT="train[:10000]"

# Extract base model name for naming
BASE_MODEL_NAME=$(basename "$BASE_MODEL" | tr '/' '_' | tr '-' '_')

# Training parameters
LEARNING_RATE="1e-4"
BATCH_SIZE=4
GRAD_ACCUM=1
NUM_EPOCHS=2
MAX_LENGTH=2048

# Method-specific parameters
LORA_R=4
QALORA_GROUP_SIZE=32
PISSA_NITER=4

# Other training parameters
WARMUP_RATIO=0.03
LR_SCHEDULER_TYPE="cosine"
LOGGING_STEPS=10
SAVE_STEPS=25000
EVAL_STEPS=500
BF16="True"
DATALOADER_PIN_MEMORY="False"
REMOVE_UNUSED_COLUMNS="False"
REPORT_TO="none"

# Evaluation parameters
EVAL_TASKS="hellaswag,piqa,winogrande,arc_easy,arc_challenge,boolq,openbookqa,wikitext"
NUM_FEWSHOT=5
EVAL_BATCH_SIZE=1
EVAL_LIMIT=300

# Dataset name for paths
DATASET_NAME=$(basename "$DATASET" | sed 's/-/_/g')

# Define all training modes to run (using space-separated string instead of array)
TRAINING_MODES="full lora qlora qalora pissa corda"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to run a single experiment
run_experiment() {
    local training_mode=$1
    
    # Generate experiment name with base model
    case "$training_mode" in
        "qalora")
            experiment_name="${BASE_MODEL_NAME}_${DATASET_NAME}_${training_mode}_r_${LORA_R}_group_${QALORA_GROUP_SIZE}"
            ;;
        "pissa")
            experiment_name="${BASE_MODEL_NAME}_${DATASET_NAME}_${training_mode}_r_${LORA_R}_niter_${PISSA_NITER}"
            ;;
        "lora"|"corda")
            experiment_name="${BASE_MODEL_NAME}_${DATASET_NAME}_${training_mode}_r_${LORA_R}"
            ;;
        *)
            experiment_name="${BASE_MODEL_NAME}_${DATASET_NAME}_${training_mode}"
            ;;
    esac
    
    output_dir="train_results_${experiment_name}"
    eval_dir="eval_results_${experiment_name}"
    
    echo -e "${BLUE}🚀 Starting experiment: $experiment_name${NC}"
    echo -e "${BLUE}  Base model: $BASE_MODEL${NC}"
    echo -e "${BLUE}  Training mode: $training_mode${NC}"
    echo -e "${BLUE}  Output dir: $output_dir${NC}"
    echo -e "${BLUE}  Eval dir: $eval_dir${NC}"
    echo -e "${BLUE}  Dataset: $DATASET${NC}"
    echo -e "${BLUE}  LoRA rank: $LORA_R${NC}"
    
    if [ "$training_mode" = "qalora" ]; then
        echo -e "${BLUE}  QALoRA group size: $QALORA_GROUP_SIZE${NC}"
    elif [ "$training_mode" = "pissa" ]; then
        echo -e "${BLUE}  PiSSA iterations: $PISSA_NITER${NC}"
    fi
    
    echo ""
    
    # Phase 1: Training
    echo -e "${YELLOW}📚 Phase 1: Training with $training_mode...${NC}"
    
    python corda_finetuning.py \
        --model_name_or_path="$BASE_MODEL" \
        --training_mode="$training_mode" \
        --lora_r="$LORA_R" \
        --qalora_group_size="$QALORA_GROUP_SIZE" \
        --pissa_niter="$PISSA_NITER" \
        --learning_rate="$LEARNING_RATE" \
        --per_device_train_batch_size="$BATCH_SIZE" \
        --data_path="$DATASET" \
        --dataset_split="$DATASET_SPLIT" \
        --dataset_field "instruction" "output" \
        --num_train_epochs="$NUM_EPOCHS" \
        --output_dir="$output_dir" \
        --model_max_length="$MAX_LENGTH" \
        --gradient_accumulation_steps="$GRAD_ACCUM" \
        --warmup_ratio="$WARMUP_RATIO" \
        --lr_scheduler_type="$LR_SCHEDULER_TYPE" \
        --logging_steps="$LOGGING_STEPS" \
        --save_steps="$SAVE_STEPS" \
        --eval_steps="$EVAL_STEPS" \
        --bf16="$BF16" \
        --dataloader_pin_memory="$DATALOADER_PIN_MEMORY" \
        --remove_unused_columns="$REMOVE_UNUSED_COLUMNS" \
        --report_to="$REPORT_TO"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Training failed for $training_mode!${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Training completed for $training_mode!${NC}"
    
    # Wait for cleanup
    sleep 3
    
    # Phase 2: Evaluation
    echo -e "${YELLOW}📊 Phase 2: Evaluation for $training_mode...${NC}"
    
    python eval_peft.py \
        --model_name_or_path="$output_dir/ft" \
        --base_model="$BASE_MODEL" \
        --tasks="$EVAL_TASKS" \
        --num_fewshot="$NUM_FEWSHOT" \
        --per_device_eval_batch_size="$EVAL_BATCH_SIZE" \
        --test_generation \
        --output_dir="$eval_dir" \
        --limit="$EVAL_LIMIT"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Evaluation failed for $training_mode!${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✅ Experiment completed for $training_mode!${NC}"
    echo -e "${GREEN}📁 Training results saved in: $output_dir${NC}"
    echo -e "${GREEN}📁 Evaluation results saved in: $eval_dir${NC}"
    echo ""
    
    # Optional: Run with post-quantization for comparison
    if [ "$training_mode" != "full" ]; then
        echo -e "${YELLOW}🔧 Phase 3: Evaluation with post-quantization for $training_mode...${NC}"
        
        eval_dir_quant="${eval_dir}_post_quant"
        
        python eval_peft.py \
            --model_name_or_path="$output_dir/ft" \
            --base_model="$BASE_MODEL" \
            --tasks="$EVAL_TASKS" \
            --num_fewshot="$NUM_FEWSHOT" \
            --per_device_eval_batch_size="$EVAL_BATCH_SIZE" \
            --output_dir="$eval_dir_quant" \
            --limit="$EVAL_LIMIT" \
            --apply_gptq_post_quant
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Post-quantization evaluation completed for $training_mode!${NC}"
            echo -e "${GREEN}📁 Post-quant eval results saved in: $eval_dir_quant${NC}"
        else
            echo -e "${YELLOW}⚠️  Post-quantization evaluation failed for $training_mode (continuing...)${NC}"
        fi
    fi
    
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

# Function to create summary report
create_summary() {
    echo -e "${BLUE}📈 Creating experiment summary...${NC}"
    
    summary_file="experiment_summary_${BASE_MODEL_NAME}_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$summary_file" << EOF
# Experiment Summary

**Base Model:** $BASE_MODEL  
**Dataset:** $DATASET  
**Dataset Split:** $DATASET_SPLIT  
**Date:** $(date)

## Configuration
- Learning Rate: $LEARNING_RATE
- Batch Size: $BATCH_SIZE
- Epochs: $NUM_EPOCHS
- Max Length: $MAX_LENGTH
- LoRA Rank: $LORA_R
- QALoRA Group Size: $QALORA_GROUP_SIZE
- PiSSA Iterations: $PISSA_NITER

## Training Modes Tested
EOF
    
    for mode in $TRAINING_MODES; do
        case "$mode" in
            "qalora")
                experiment_name="${BASE_MODEL_NAME}_${DATASET_NAME}_${mode}_r_${LORA_R}_group_${QALORA_GROUP_SIZE}"
                ;;
            "pissa")
                experiment_name="${BASE_MODEL_NAME}_${DATASET_NAME}_${mode}_r_${LORA_R}_niter_${PISSA_NITER}"
                ;;
            "lora"|"corda")
                experiment_name="${BASE_MODEL_NAME}_${DATASET_NAME}_${mode}_r_${LORA_R}"
                ;;
            *)
                experiment_name="${BASE_MODEL_NAME}_${DATASET_NAME}_${mode}"
                ;;
        esac
        
        echo "- **$mode**: train_results_$experiment_name" >> "$summary_file"
    done
    
    echo "" >> "$summary_file"
    echo "## Evaluation Tasks" >> "$summary_file"
    echo "$EVAL_TASKS" | tr ',' '\n' | sed 's/^/- /' >> "$summary_file"
    
    echo -e "${GREEN}📄 Summary saved to: $summary_file${NC}"
}

# Main execution
echo -e "${BLUE}🔬 Starting comprehensive training experiment${NC}"
echo -e "${BLUE}Base Model: $BASE_MODEL${NC}"
echo -e "${BLUE}Training Modes: $TRAINING_MODES${NC}"
echo -e "${BLUE}Dataset: $DATASET${NC}"
echo ""

# Track successful and failed experiments
successful_experiments=""
failed_experiments=""

# Run all experiments
for mode in $TRAINING_MODES; do
    if run_experiment "$mode"; then
        successful_experiments="$successful_experiments $mode"
    else
        failed_experiments="$failed_experiments $mode"
        echo -e "${RED}❌ Experiment failed for $mode, continuing with next...${NC}"
    fi
    
    # Clean up GPU memory between experiments
    echo -e "${YELLOW}🧹 Cleaning up GPU memory...${NC}"
    python -c "import torch; torch.cuda.empty_cache(); print('GPU memory cleared')" 2>/dev/null || true
    sleep 5
done

# Final summary
echo -e "${BLUE}🏁 All experiments completed!${NC}"
echo ""

# Count successful experiments
success_count=$(echo $successful_experiments | wc -w)
echo -e "${GREEN}✅ Successful experiments ($success_count):${NC}"
for exp in $successful_experiments; do
    echo -e "${GREEN}  - $exp${NC}"
done

# Count failed experiments
fail_count=$(echo $failed_experiments | wc -w)
if [ $fail_count -gt 0 ]; then
    echo ""
    echo -e "${RED}❌ Failed experiments ($fail_count):${NC}"
    for exp in $failed_experiments; do
        echo -e "${RED}  - $exp${NC}"
    done
fi

# Create summary report
create_summary

echo ""
echo -e "${BLUE}🎉 Comprehensive experiment completed!${NC}"
echo -e "${BLUE}Check individual result directories for detailed outputs.${NC}"