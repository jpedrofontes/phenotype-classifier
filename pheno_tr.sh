#!/bin/bash

#SBATCH --job-name=pheno_tr
#SBATCH --time=1000:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --nodelist=vision2
#SBATCH --gres=gpu:1
#SBATCH --mem=122000
#SBATCH --cpus-per-task=32
#SBATCH -o /home/mguevaral/jpedro/phenotype-classifier/logs/%x.%j.out 

export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_CPP_MIN_LOG_LEVEL=1
source /home/mguevaral/jpedro/phenotype-classifier/venv/bin/activate
module load CUDA
module load cuDNN

TRAIN_AUTOENCODER=false
TUNE_AUTOENCODER=false

while getopts "at" opt; do
  case $opt in
    a)
      TRAIN_AUTOENCODER=true
      ;;
    t)
      TUNE_AUTOENCODER=true
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

if [ "$TRAIN_AUTOENCODER" = true ]; then
  if [ "$TUNE_AUTOENCODER" = true ]; then
    # Train the autoencoder model with hyperparameter tuning
    python /home/mguevaral/jpedro/phenotype-classifier/train_and_evaluate.py --autoencoder --tune
  else
    # Train the autoencoder model without hyperparameter tuning
    python /home/mguevaral/jpedro/phenotype-classifier/train_and_evaluate.py --autoencoder
  fi
fi

# Run the model for random forest and SVM for each phenotype
PHENOTYPES=(0 1 2 3)
PHENOTYPE_NAMES=("Luminal A" "Luminal B" "HER2 Enriched" "Triple Negative")

for PHENOTYPE in "${PHENOTYPES[@]}"; do
    PHENOTYPE_NAME=${PHENOTYPE_NAMES[$PHENOTYPE]}
    printf "\nTraining and evaluating SVM for phenotype $PHENOTYPE ($PHENOTYPE_NAME)\n"
    python /home/mguevaral/jpedro/phenotype-classifier/train_and_evaluate.py --phenotype $PHENOTYPE --svm
done
