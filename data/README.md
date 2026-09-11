# Dataset

This project uses the HAM10000 dataset:

Tschandl, P., Rosendahl, C., & Kittler, H.  
*The HAM10000 dataset: A large collection of multi-source dermatoscopic images of common pigmented skin lesions.*  
Scientific Data, 5, 180161 (2018).

The dataset itself is not included in this repository.

## Dataset characteristics

- 10,015 dermoscopic images
- 7,470 unique skin lesions
- 7 diagnostic categories

The diagnostic classes are:

- `akiec` — Actinic keratoses and intraepithelial carcinoma
- `bcc` — Basal cell carcinoma
- `bkl` — Benign keratosis-like lesions
- `df` — Dermatofibroma
- `mel` — Melanoma
- `nv` — Melanocytic nevi
- `vasc` — Vascular lesions

## Local dataset

The processed dataset used by the notebook should be stored locally as:

data/HAM10000_data.pkl

This file is excluded from the Git repository.