# Evaluating Fairness and Bias in Pretrained Facial Recognition Models

This repository contains the implementation, experiments, and results for the MSc Software Engineering final project at the University of Hertfordshire.  
The project investigates **fairness in pretrained facial recognition models** using two widely studied datasets:

- **FairFace**: Balanced across race, gender, and age.
- **UTKFace**: Naturally imbalanced but widely used.

The FaceNet architecture was extended with **multi-task classification heads** for race, gender, and age prediction, and evaluated with **model-agnostic interpretability tools** (SHAP, LIME, Grad-CAM).

---

## 📖 Project Overview

Facial recognition technology has promising applications in security, healthcare, and social contexts, but suffers from persistent algorithmic bias.  
This project:

- Profiles bias in **FairFace** and **UTKFace** datasets.
- Evaluates model performance across **age, gender, and race** subgroups.
- Applies **interpretability methods** to explain bias mechanisms.
- Proposes a reproducible **framework** for fairness evaluation.

📄 Full details are in the [Final Project Report](report/FR_project.pdf).

---

## 📂 Repository Structure

- `notebooks/` – Jupyter notebooks for FairFace and UTKFace experiments.
- `data/` – Instructions for downloading datasets (datasets not included).
- `results/` – Saved figures, logs, and evaluation outputs.
- `report/` – MSc final project report (PDF).

---

## ⚙️ Setup & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/face-recognition-model.git
   cd face-recognition-model

**create and use a virtual environment

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

---
## Acknowledgements
Supervisor: Dr. Muhammad Yaqoob

University of Hertfordshire – School of Physics, Engineering and Computer Science

Datasets: FairFace, UTKFace

Tools: TensorFlow, Keras, SHAP, LIME, Grad-CAM

