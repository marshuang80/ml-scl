from pathlib import Path

# Directories 
PROJECT_DIR = Path("temp")   # need to modify
if not PROJECT_DIR.is_dir():
    raise Exception("Please modify PROJECT_DATA_DIR in constants to a valid directory")

# Project folders 
PROJECT_DATA_DIR = PROJECT_DIR / "chexpert"
CHEXPERT_DATA_DIR = PROJECT_DATA_DIR / "CheXpert" 
CHEXPERT_V1_DIR = CHEXPERT_DATA_DIR / "CheXpert-v1.0"

# Project cvs files 
CHEXPERT_TRAIN_CSV = CHEXPERT_V1_DIR / "train.csv"
CHEXPERT_VALID_CSV = CHEXPERT_V1_DIR / "valid.csv"

# Project image folders
CHEXPERT_TRAIN_DATA = CHEXPERT_V1_DIR / "train"
CHEXPERT_VALID_DATA = CHEXPERT_V1_DIR / "valid"

# ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Chexpert labels
CHEXPERT_TASKS = ["No Finding",
                  "Enlarged Cardiomediastinum",
                  "Cardiomegaly",
                  "Lung Lesion",
                  "Airspace Opacity",
                  "Edema",
                  "Consolidation",
                  "Pneumonia",
                  "Atelectasis",
                  "Pneumothorax",
                  "Pleural Effusion",
                  "Pleural Other",
                  "Fracture",
                  "Support Devices"]

# CheXpert competition labels
CHEXPERT_COMPETITION_TASKS = ["Atelectasis",
                              "Cardiomegaly",
                              "Consolidation",
                              "Edema",
                              "Pleural Effusion"]