# ViT_MPM
## Benchmarking Vision Transformers and Convolutional Neural Networks for Mineral Prospectivity Mapping Using Sentinel-2 Imagery
### Case Study: Reko Diq Porphyry Cu–Au Deposit, Pakistan

This repository contains code, preprocessing scripts, and supporting material for my research work:

**Khan, F., & Baig, S. (2025). Benchmarking Vision Transformers and Convolutional Neural Networks for Porphyry Cu–Au Prospectivity Mapping Using Sentinel-2 Imagery: The Reko Diq Case Study**

---

## 📁 Repository Structure

ViT_MPM/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── Script/
│   ├── Preprocessing.py
│   ├── cnn_core.py              (9 MSI bands)
│   ├── vit_core.py              (9 MSI bands)
│   ├── plots.py      
│
├── data/
│   └── README.md                (data not uploaded — too large)
│
└── results/
    ├── README.md
    ├── Figures/
    └── Maps/
       
---

## ▶️ How to Run This Project (Anaconda Environment)

### 1. Install required libraries:
pip install -r requirements.txt


### 2. Run preprocessing:
python preprocessing/Preprocessing.py

### 3. Train and evaluate models:

**MSI-based CNN**
python models/cnn_core.py

**MSI-based ViT**
python models/vit_core.py


---

## 📦 Data Availability
Full Sentinel-2 data and generated training tiles are too large for GitHub.  
You can download Sentinel-2 imagery from USGS Earth Explorer:  
https://earthexplorer.usgs.gov/

Use the preprocessing script to recreate all training tiles.

---

## 📝 Citation
Please cite this work as:

**Khan, F., & Baig, S. (2025). Benchmarking Vision Transformers and Convolutional Neural Networks for Porphyry Cu–Au Prospectivity Mapping Using Sentinel-2 Imagery: The Reko Diq Case Study.**

---

## 📧 Contact
Author: **Feroz Khan**  
Email: **feroz_hallian@hotmail.com**

