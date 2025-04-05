#!/bin/bash

# Script to run the NervformerV2 demo which shows its superior performance
# This script will train and evaluate the NervformerV2 model on a single subject

# Ensure the conda environment is activated
source ~/anaconda3/etc/profile.d/conda.sh
conda activate BCI

# Set the subject ID (default: sub-01)
SUBJECT=${1:-"sub-01"}
echo "Running NervformerV2 demo for subject: $SUBJECT"

# Step 1: Train the NervformerV2 model for feature extraction
echo "Step 1: Training NervformerV2 model for EEG feature extraction"
cd Retrieval
python NervformerV2_insubject_retrieval.py --subject $SUBJECT --epochs 50 --batch_size 1024

# Step 2: Extract EEG features using the trained model
echo "Step 2: Extracting EEG features using NervformerV2"
cd ../Generation
python NervformerV2_insubject_retrival_All_extracted_feature_step1.py --subject $SUBJECT

# Step 3: Generate images from EEG features
echo "Step 3: Generating images from EEG features"
python NervformerV2_insubject_retrival_All_train_dfs_step2.py --subject $SUBJECT --num_images 10

# Step 4: Compute evaluation metrics
echo "Step 4: Computing evaluation metrics"
cd fMRI-reconstruction-NSD/src
python compute_metrics.py --subject $SUBJECT --model NervformerV2

echo "Demo completed! Results can be found in:"
echo "- Generated images: Generation/gen_images/"
echo "- Metrics: Generation/fMRI-reconstruction-NSD/src/metrics_results.csv"

echo "To compare with other models, run the following scripts:"
echo "- bash run_comparison.sh $SUBJECT" 