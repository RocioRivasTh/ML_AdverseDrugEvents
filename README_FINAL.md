
# ML_AdverseDrugReactions

## 🧠 Project Overview

This project applies supervised machine learning techniques to predict adverse drug reactions (ADRs) from the chemical structure of pharmaceutical compounds. We use a public dataset from Kaggle and enrich it with canonical SMILES generated using RDKit in Google Colab.

## 💾 Dataset

- **Source**: [Kaggle – Dataset for Adverse Drug Reaction](https://www.kaggle.com/datasets/qwertychuahn/dataset-for-adverse-drug-reaction?select=adr_dataset.csv)
- **File Used**: `data_sample_enriched.csv`
- **Description**:
    - Each row represents a drug compound using SMILES notation
    - 24 binary columns represent different types of adverse drug reactions (ADR)
    - The dataset has been enriched with canonical SMILES using the RDKit library

## 🎯 Goal

To build a machine learning model capable of:
- Predicting the types of ADRs associated with a compound
- Optionally clustering compounds by similar ADR profiles

## 📂 Repository Structure

```
src/
│
├── data_sample/               # Final enriched dataset (CSV < 5MB)
├── img/                       # EDA graphs and visualizations
├── notebooks/                 # Development notebooks (exploration, feature engineering)
├── results_notebook/          # Final notebook for reproducible execution
├── models/                    # Trained models saved as .joblib or .pkl
└── utils/                     # Custom functions used in pipeline
```

## 🧪 Methods Used

- Exploratory Data Analysis (EDA)
- Multilabel Classification (RandomForest, OneVsRest)
- (Optional) Clustering using KMeans
- Evaluation Metrics: Accuracy, F1-score, Silhouette Score (if clustering)

## 📌 Notes

- All modeling and processing is performed in Python
- Canonical SMILES were computed using RDKit from Google Colab
- Dataset enriched and ready for downstream Machine Learning tasks
