import os

base_data_dir = "/data/mguevaral/jpedro"
batch_size = 32
dataset_dir = "/data/mguevaral/crop_bbox/"
input_size = (64, 128, 128)
job_id = os.environ.get("SLURM_JOB_ID")
num_epochs = 1000
phenotypes = {0: "Luminal_A", 1: "Luminal_B", 2: "HER2_Enriched", 3: "Triple_Negative"}
random_seed = 42 
