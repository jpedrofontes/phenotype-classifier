#!/bin/sh

#SBATCH --job-name=phenotype_classifier_train
#SBATCH --time=1000:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --nodelist=vision1
#SBATCH --gres=gpu:2
#SBATCH --mem=122000
#SBATCH --cpus-per-task=64
#SBATCH -o /home/mguevaral/jpedro/phenotype-classifier/logs/%x.%j.out 

export TF_GPU_ALLOCATOR=cuda_malloc_async
source /home/mguevaral/jpedro/phenotype-classifier/venv/bin/activate
module load CUDA
module load cuDNN

python /home/mguevaral/jpedro/phenotype-classifier/cross_validation.py -p $1 -k $2 
