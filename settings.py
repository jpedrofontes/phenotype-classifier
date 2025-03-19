import os

class Settings:
    # Directories
    BASE_DATA_DIR: str = "/data/mguevaral/jpedro"
    DATASET_DIR: str = "/data/mguevaral/crop_bbox/"
    
    # Slurm stuff 
    JOB_ID: str = os.environ.get("SLURM_JOB_ID")
    
    # Data
    PHENOTYPES: dict = {0: "Luminal_A", 1: "Luminal_B", 2: "HER2_Enriched", 3: "Triple_Negative"}
    
    # Training
    BATCH_SIZE: int = 2
    INPUT_SIZE: tuple = (64, 128, 128)
    NUM_EPOCHS: int = 1000
    
    # Reproducibility
    RANDOM_SEED: int = 42 
