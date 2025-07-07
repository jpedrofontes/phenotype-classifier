import os
from typing import Dict, Tuple

class Settings:
    # Directories
    BASE_DATA_DIR: str = "/data/mguevaral/jpedro"
    DUKE_DATASET_DIR: str = "/data/mguevaral/crop_bbox/"
    ADVA_DATASET_DIR: str = "/data/evanfonseca/BC-Collection-Phenotyping/OutputJPG1"
    ADVA_CSV_PATH: str = (
        "/home/evanfonseca/BC-Collection-Phenotyping/Detections_Export/result_detections_ADVA.csv"
    )
    ISPY1_DATASET_DIR: str = "/data/evanfonseca/BC-Collection-Phenotyping/OutputJPG1"
    ISPY1_CSV_PATH: str = (
        "/home/evanfonseca/BC-Collection-Phenotyping/Detections_Export/result_detections_ISPY1.csv"
    )
    ISPY2_DATASET_DIR: str = "/data/evanfonseca/BC-Collection-Phenotyping/OutputJPG1"
    ISPY2_CSV_PATH: str = (
        "/home/evanfonseca/BC-Collection-Phenotyping/Detections_Export/result_detections_ISPY2.csv"
    )
    SETUBAL_DATASET_DIR: str = "/data/mguevaral/jpedro/setubal-dataset/Detection_Export"
    SETUBAL_CSV_PATH: str = (
        "/data/mguevaral/jpedro/setubal-dataset/Detection_Export_Setubal.csv"
    )

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
    CROP_SIZE: Tuple[int, int, int] = (128, 128)
    NUM_EPOCHS: int = 1000

    # Reproducibility
    RANDOM_SEED: int = 42 
