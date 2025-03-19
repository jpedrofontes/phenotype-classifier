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
TRAIN_CNN=false
TRAIN_RESNET=false
TRAIN_SVM=false

while getopts "atcrs" opt; do
  case $opt in
    a)
      TRAIN_AUTOENCODER=true
      ;;
    t)
      TUNE_AUTOENCODER=true
      ;;
    c)
      TRAIN_CNN=true
      ;;
    r)
      TRAIN_RESNET=true
      ;;
    s)
      TRAIN_SVM=true
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

  # Run the model for SVM for each phenotype
  PHENOTYPES=(0 1 2 3)
  PHENOTYPE_NAMES=("Luminal A" "Luminal B" "HER2 Enriched" "Triple Negative")

  for PHENOTYPE in "${PHENOTYPES[@]}"; do
      PHENOTYPE_NAME=${PHENOTYPE_NAMES[$PHENOTYPE]}
      printf "\nTraining and evaluating SVM for phenotype $PHENOTYPE ($PHENOTYPE_NAME)\n"
      python /home/mguevaral/jpedro/phenotype-classifier/train_and_evaluate.py --phenotype $PHENOTYPE --svm
  done
elif [ "$TRAIN_SVM" = true ]; then
  # Run the model for SVM for each phenotype
  PHENOTYPES=(0 1 2 3)
  PHENOTYPE_NAMES=("Luminal A" "Luminal B" "HER2 Enriched" "Triple Negative")
  CSV_FILE_PATH="/home/mguevaral/jpedro/phenotype-classifier/datasets/latent_space_values.csv"

  for PHENOTYPE in "${PHENOTYPES[@]}"; do
      PHENOTYPE_NAME=${PHENOTYPE_NAMES[$PHENOTYPE]}
      printf "\nTraining and evaluating SVM for phenotype $PHENOTYPE ($PHENOTYPE_NAME)\n"
      python /home/mguevaral/jpedro/phenotype-classifier/train_and_evaluate.py --phenotype $PHENOTYPE --svm -csv $CSV_FILE_PATH
  done
elif [ "$TRAIN_CNN" = true ]; then
  # Run the model for CNN for each phenotype
  PHENOTYPES=(0 1 2 3)
  PHENOTYPE_NAMES=("Luminal A" "Luminal B" "HER2 Enriched" "Triple Negative")

  for PHENOTYPE in "${PHENOTYPES[@]}"; do
      PHENOTYPE_NAME=${PHENOTYPE_NAMES[$PHENOTYPE]}
      printf "\nTraining and evaluating CNN for phenotype $PHENOTYPE ($PHENOTYPE_NAME)\n"
      python /home/mguevaral/jpedro/phenotype-classifier/train_and_evaluate.py --phenotype $PHENOTYPE
  done
elif [ "$TRAIN_RESNET" = true ]; then
  # Run the model for ResNet for each phenotype
  PHENOTYPES=(0 1 2 3)
  PHENOTYPE_NAMES=("Luminal A" "Luminal B" "HER2 Enriched" "Triple Negative")

  for PHENOTYPE in "${PHENOTYPES[@]}"; do
      PHENOTYPE_NAME=${PHENOTYPE_NAMES[$PHENOTYPE]}
      printf "\nTraining and evaluating ResNet for phenotype $PHENOTYPE ($PHENOTYPE_NAME)\n"
      python /home/mguevaral/jpedro/phenotype-classifier/train_and_evaluate.py --phenotype $PHENOTYPE --resnet
  done
fi
