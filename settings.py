import os
from typing import Dict, Tuple

class Settings:
    # Directories
    BASE_DATA_DIR: str = "/data/mguevaral/jpedro"
    DATASET_DIR: str = "/data/mguevaral/crop_bbox/"
    
    # Slurm stuff 
    JOB_ID: str = os.environ.get("SLURM_JOB_ID")
    
    PHENOTYPES: Dict[int, str] = {
        0: "Luminal A", 
        1: "Luminal B", 
        2: "HER2 Enriched", 
        3: "Triple Negative"
    }
    
    # Training
    BATCH_SIZE: int = 2
    INPUT_SIZE: Tuple[int, int, int] = (64, 128, 128)
    NUM_EPOCHS: int = 1000
    
    # Reproducibility
    RANDOM_SEED: int = 42 
