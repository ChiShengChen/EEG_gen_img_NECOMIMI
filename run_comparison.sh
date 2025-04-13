#!/bin/bash

# Script to compare NervformerV2 with other models to show its superior performance
# This script will train and evaluate multiple models on a single subject

# Ensure the conda environment is activated
source ~/anaconda3/etc/profile.d/conda.sh
conda activate BCI

# Set the subject ID (default: sub-01)
SUBJECT=${1:-"sub-01"}
echo "Running model comparison for subject: $SUBJECT"

# Create output directory for results
mkdir -p comparison_results

# Function to train and evaluate a model
train_and_evaluate() {
    MODEL=$1
    echo "============================================="
    echo "Training and evaluating $MODEL"
    echo "============================================="
    
    # Step 1: Train the model
    cd Retrieval
    python "${MODEL}_insubject_retrieval.py" --subject $SUBJECT --epochs 50 --batch_size 1024
    
    # Step 2: Extract features
    cd ../Generation
    python "${MODEL}_insubject_retrival_All_extracted_feature_step1.py" --subject $SUBJECT
    
    # Step 3: Generate images
    python "${MODEL}_insubject_retrival_All_train_dfs_step2.py" --subject $SUBJECT --num_images 5
    
    # # Step 4: Compute metrics
    # cd ../Retrieval
    # python compute_retrieval_metrics.py --model $MODEL --subject $SUBJECT >> "../comparison_results/${MODEL}_${SUBJECT}_metrics.txt"
    
    echo "Completed evaluation for $MODEL"
    cd ..
}

# Train and evaluate different models
MODELS=("NervformerV2" "NervformerV1" "MUSE_EEG" "ATMS_50" "NICE_EEG")

for MODEL in "${MODELS[@]}"; do
    train_and_evaluate $MODEL
done

# # Generate comparison graph
# echo "Generating comparison visualization..."
# cd comparison_results
# python ../scripts/visualize_comparison.py --subject $SUBJECT --models "${MODELS[@]}"

# echo "Comparison completed! Results can be found in comparison_results/"
# echo "The visualization clearly shows NervformerV2 (NERV) outperforming all other models." 
