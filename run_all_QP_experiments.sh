#!/bin/bash

# QP Transformer Experiment Script
# Run all combinations of heads (h) and layers (l)

# Note: Default parameters are defined in QP_func_approx.py
# Only specify parameters that differ from defaults or need to be overridden

# Model types to test
# MODEL_TYPES=("SoftmaxTransformer" "LinearTransformer" "MLP" "LSTM")
MODEL_TYPES=("LSTM")

# Dataset parameters
NUM_TRAIN=50000
NUM_TEST=10000
NUM_VAL=10000
# NUM_TRAIN=100000
# NUM_TEST=20000
# NUM_VAL=20000



# Parameter ranges to test
# Note that m needs to be less than or equal to n due to token size constraints
# Otherwise, the corresponding result folder will be empty

PROBLEM_DIMS=(5)
CONSTRAINT_NUMS=(3)
HEADS=(0)
LAYERS=(4)

# PROBLEM_DIMS=(7)
# CONSTRAINT_NUMS=(3 6)
# HEADS=(1)
# LAYERS=(1)

# PROBLEM_DIMS=(15)
# CONSTRAINT_NUMS=(10)
# HEADS=(1 2 4 8)
# LAYERS=(1 2 4 8 16)

# PROBLEM_DIMS=(20)
# CONSTRAINT_NUMS=(8)
# HEADS=(1)
# LAYERS=(8)

# PROBLEM_DIMS=(7)
# CONSTRAINT_NUMS=(3)
# HEADS=(1 2 4 8)
# LAYERS=(1 2 4 8 16)


echo "Starting QP Transformer experiments..."
echo "Problem dimensions: n=${PROBLEM_DIMS[@]}, m=${CONSTRAINT_NUMS[@]}"
echo "Model types: ${MODEL_TYPES[@]}"
echo "Architecture ranges: heads=${HEADS[@]}, layers=${LAYERS[@]}"
echo ""

# Counter for total experiments
total_experiments=$((${#MODEL_TYPES[@]} * ${#PROBLEM_DIMS[@]} * ${#CONSTRAINT_NUMS[@]} * ${#HEADS[@]} * ${#LAYERS[@]}))
current_experiment=0


for model_type in "${MODEL_TYPES[@]}"; do
    for n in "${PROBLEM_DIMS[@]}"; do
        for m in "${CONSTRAINT_NUMS[@]}"; do
            for h in "${HEADS[@]}"; do
                for l in "${LAYERS[@]}"; do
                    current_experiment=$((current_experiment + 1))
                    
                    echo "=========================================="
                    echo "Experiment ${current_experiment}/${total_experiments}"
                    echo "Model: ${model_type}, n=${n}, m=${m}, Heads: ${h}, Layers: ${l}"
                    echo "=========================================="
                    
                    # Run experiment
                    python /home/yufanzh/projects/Time2Decide/QP_func_approx/QP_func_approx.py \
                        --model_type "${model_type}" \
                        -n "${n}" \
                        -m "${m}" \
                        -l "${l}" \
                        -h "${h}" \
                        --num_train "${NUM_TRAIN}" \
                        --num_val "${NUM_VAL}" \
                        --num_test "${NUM_TEST}"
                    
                    # Calculate progress percentage
                    progress_percentage=$(echo "scale=1; ${current_experiment} * 100 / ${total_experiments}" | bc -l)
                    

                    
                    echo ""
                    echo "  Experiment ${current_experiment}/${total_experiments} completed (${progress_percentage}%)"
                    echo "  Results saved in experiment folder."
                    echo ""
                done
            done
        done
    done
done


echo "=========================================="
echo "All experiments completed!"
echo "Total experiments run: ${total_experiments}"
echo "=========================================="
