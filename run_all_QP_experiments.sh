#!/bin/bash

# QP Transformer Experiment Script
# Run all combinations of heads (h) and layers (l)

# Note: Default parameters are defined in QP_func_approx.py
# Only specify parameters that differ from defaults or need to be overridden

# Model types to test
# MODEL_TYPES=("SoftmaxTransformer" "LinearTransformer" "MLP")
MODEL_TYPES=("LinearTransformer" "SoftmaxTransformer")


# Parameter ranges to test
# Note that m needs to be less than or equal to n due to token size constraints
# Otherwise, the corresponding result folder will be empty


# PROBLEM_DIMS=(5)
# CONSTRAINT_NUMS=(3)
# HEADS=(1 2 4 8)
# LAYERS=(1 2 4 8 16)

PROBLEM_DIMS=(10)
CONSTRAINT_NUMS=(9)
HEADS=(1 2 4 8)
LAYERS=(1 2 4 8 16)

# PROBLEM_DIMS=(10)
# CONSTRAINT_NUMS=(6)
# HEADS=(1 2 4 8)
# LAYERS=(1 2 4 8 16)

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

# Start time for total experiments
total_start_time=$(date +%s)

for model_type in "${MODEL_TYPES[@]}"; do
    for n in "${PROBLEM_DIMS[@]}"; do
        for m in "${CONSTRAINT_NUMS[@]}"; do
            for h in "${HEADS[@]}"; do
                for l in "${LAYERS[@]}"; do
                    current_experiment=$((current_experiment + 1))
                    
                    # Start time for current experiment
                    exp_start_time=$(date +%s)
                    
                    echo "=========================================="
                    echo "Experiment ${current_experiment}/${total_experiments}"
                    echo "Model: ${model_type}, n=${n}, m=${m}, Heads: ${h}, Layers: ${l}"
                    echo "=========================================="
                    
                    # Run experiment
                    python QP_func_approx.py \
                        --model_type "${model_type}" \
                        -n "${n}" \
                        -m "${m}" \
                        -l "${l}" \
                        -h "${h}"
                    
                    # End time for current experiment
                    exp_end_time=$(date +%s)
                    exp_duration=$((exp_end_time - exp_start_time))
                    
                    # Calculate progress percentage
                    progress_percentage=$(echo "scale=1; ${current_experiment} * 100 / ${total_experiments}" | bc -l)
                    
                    # Format experiment duration
                    exp_hours=$((exp_duration / 3600))
                    exp_minutes=$(((exp_duration % 3600) / 60))
                    exp_seconds=$((exp_duration % 60))
                    
                    if [ ${exp_hours} -gt 0 ]; then
                        exp_duration_str="${exp_hours}h ${exp_minutes}m ${exp_seconds}s"
                    elif [ ${exp_minutes} -gt 0 ]; then
                        exp_duration_str="${exp_minutes}m ${exp_seconds}s"
                    else
                        exp_duration_str="${exp_seconds}s"
                    fi
                    
                    echo ""
                    echo "✓ Experiment ${current_experiment}/${total_experiments} completed (${progress_percentage}%)"
                    echo "  Duration: ${exp_duration_str}"
                    echo "  Results saved in experiment folder."
                    echo ""
                done
            done
        done
    done
done

# Calculate total duration
total_end_time=$(date +%s)
total_duration=$((total_end_time - total_start_time))
total_hours=$((total_duration / 3600))
total_minutes=$(((total_duration % 3600) / 60))
total_seconds=$((total_duration % 60))

if [ ${total_hours} -gt 0 ]; then
    total_duration_str="${total_hours}h ${total_minutes}m ${total_seconds}s"
elif [ ${total_minutes} -gt 0 ]; then
    total_duration_str="${total_minutes}m ${total_seconds}s"
else
    total_duration_str="${total_seconds}s"
fi

echo "=========================================="
echo "All experiments completed!"
echo "Total experiments run: ${total_experiments}"
echo "Total duration: ${total_duration_str}"
echo "=========================================="
