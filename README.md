# ML Pipeline for Predicting Exacerbation Outcome

This repository contains code and documentation for the machine learning (ML) pipeline developed to analyze and predict clinical outcomes in pediatric asthma patients. 
The models were implemented and validated as part of the study:  
**"[full study title]"**.

<p align="center">
  <img src="figures/Pipeline-v3.png" width="100%"/>
</p>

---

## 📊 Workflow Summary

The ML pipeline consists of the following key steps:

1. **Data Preparation**  
   - 80/20 train-test split on the original dataset  
   - Data preprocessing and MICE-imputation

2. **Model Training**  
   - 10-fold cross-validation on the training set  
   - Models: Logistic Regression (LR), Composite Regression (CR), Support Vector Machine (SVM), Decision Tree (XGBoost), Feedforward Neural Network (FFNN).

3. **Model Application and Evaluation**  
   - Trained models are applied to the test set  
   - Scoring includes accuracy, precision, recall, F1-score, Training vs Validation Loss, Training vs. Validation Accuracy, ROC-AUC
  
## 📌 Reproducibility

To reproduce the results:

### Create environment

`conda create -n ml-pipeline python=3.9 -y`  
`conda activate ml-pipeline`  
`pip install -r requirements.txt`

### Run training
``python src/training.py``

### Run evaluation
``python src/evaluation.py``

## 👤 Contributors
This project was created and developed by:

**Cristina Longo, PhD**  
*Project leader, University of Montreal*

**Oleg S. Matusovsky, PhD** — [GitHub: matusoff](https://github.com/matusoff)  
*Lead developer of the ML models, data pipeline, and workflow design.*

**Data contributions from:**  
Co-author 1  
Co-author 2  
Co-author 3  
Co-author 4  
Co-author 5  
Co-author 6  
Co-author 7  
Co-author 8

## 📄 Citation
If you use this codebase, please cite:
``[DOI here once available]``

---

## 📂 Repository Structure

```bash
├── data/               # Raw and processed data (if allowed)
├── figures/            # Workflow images and results
├── models/             # Saved model weights or pipelines
├── notebooks/          # Jupyter Notebooks for training, evaluation
├── src/                # Core Python scripts
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
│   └── utils.py
├── requirements.txt    # Required Python packages
├── a28e3e5d-5190...png # ML pipeline figure
└── README.md
