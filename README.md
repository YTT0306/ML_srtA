# ML_srtA
A pipeline for training and evaluating machine learning models for Staphylococcus aureus sortase A inhibitors.

## Overview
This pipeline transforms SMILES (Simplified Molecular Input Line Entry System) strings into hybrid feature vectors and trains machine learning models to classify compounds as active or inactive.

### Key Features
* **Scaffold-Aware Splitting**: Uses Bemis-Murcko scaffolds to ensure that training and testing sets do not share the same core chemical structures. This provides a realistic estimate of how the model will perform on novel chemical series.
* **Hybrid Featurization**: Combines **ECFP4 (Morgan) Fingerprints** (structural fragments) with **RDKit Physical-Chemical Descriptors** (LogP, Molecular Weight, Polar Surface Area, etc.).
* **Automated Feature Selection**: Automatically removes low-variance, high-null, and highly correlated features to optimize model performance.
* **Multi-Model Support**: Support for Random Forest, XGBoost, SVM, and KNN.

## Installation

Ensure you have a Conda or Python environment with RDKit installed:

```bash
pip install pandas numpy scikit-learn xgboost rdkit matplotlib joblib
