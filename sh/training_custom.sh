#!/bin/sh

#SBATCH --job-name=JP_PHENOTYPE_CLASSIFIER_CUSTOM
#SBATCH --time=1000:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --nodelist=vision1
#SBATCH --gres=gpu:2
#SBATCH --mem=122000
#SBATCH --cpus-per-task=32

source /home/mguevaral/jpedro/phenotype-classifier/venv/bin/activate
module load CUDA
module load cuDNN

python /home/mguevaral/jpedro/phenotype-classifier/train_and_evaluate.py --sizes $@
