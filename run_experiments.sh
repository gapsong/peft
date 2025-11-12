#!/bin/bash

# Shell script to run all experiments for qalora and qlora training modes
# and evaluate the models using evaluate_residual_models.py

set -e  # Exit on any error

echo "Starting experiments for qalora and qlora training modes..."

# Define training modes
TRAINING_MODES=("qalora" "qlora")

# Run training experiments
for mode in "${TRAINING_MODES[@]}"; do
    echo "Running training experiment with mode: $mode"
    python train.py --training_mode $mode
    
    if [ $? -eq 0 ]; then
        echo "Training completed successfully for mode: $mode"
    else
        echo "Training failed for mode: $mode"
        exit 1
    fi
done

echo "All training experiments completed successfully!"

# Run evaluation
echo "Starting model evaluation..."
python evaluate_residual_models.py

if [ $? -eq 0 ]; then
    echo "Model evaluation completed successfully!"
else
    echo "Model evaluation failed!"
    exit 1
fi

echo "All experiments and evaluations completed successfully!"