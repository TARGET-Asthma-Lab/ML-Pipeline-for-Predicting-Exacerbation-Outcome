# ML Pipeline for Predicting Exacerbation Outcome

This repository contains code and documentation for the machine learning (ML) pipeline developed to analyze and predict clinical outcomes in pediatric asthma patients. 
The models were implemented and validated as part of the study:  
**"[full study title]"**.

<p align="center">
  <img src="figures/Pipeline.png" width="700"/>
</p>

---

## 📊 Workflow Summary

The ML pipeline consists of the following key steps:

1. **Data Preparation**  
   - 80/20 train-test split on the original dataset  
   - Data preprocessing and imputation (if applicable)

2. **Model Training**  
   - 10-fold cross-validation on the training set  
   - Models: Logistic Regression (LR), Support Vector Machine (SVM), Decision Tree, Feedforward Neural Network (FFNN), etc.

3. **Model Application and Evaluation**  
   - Trained models are applied to the test set  
   - Scoring includes accuracy, precision, recall, F1-score, ROC-AUC, and others

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
