# Explainable Machine Learning for Tap Water Quality

This repository contains code for the entire pipeline used in the explainable machine learning portion of the work featured in "Explainable Machine Learning and Complex Network Theory to Manage Tap Water Quality", by Gleeson K., Kyritsakas G., Husband S., and Boxall J. (2026). 

In this work, 4 target tap water quality parameters were selected as targets, as outlined in the paper. This repository shows the pipeline used for predicting low free chlorine events (< 0.1 mg/L) at customer taps. The same code and pipeline was applied to the other 3 target features.

> **Data availability**: The underlying dataset is proprietary operational data from a UK water company and cannot be shared. The scripts are provided to document the methodology.

> **Runtime**: On a dataset of similar size to this project (~100 features, tens of thousands of samples), Step 3 (forward stepwise) is the most computationally expensive as it involves trialling every remaining candidate across 5-fold CV at each selection step amounts to hundreds of model fits per step, and can take several hours per model on a single machine. For the published work, this code was adapted to run in a parallelised form within a SLURM HPC environment; the scripts here are provided in their sequential, single-machine form for clarity.

---

## Pipeline Overview

```
All network and WQ features (~100+)
           |
           v
  01_initial_shap.py
  Each model (RF, GB, XGBoost) is trained on all features
  and its own SHAP values are computed, giving a model-specific
  importance ranking. Also performs train/test split and MICE imputation.
  --> results/shap_{model}.csv   (one per model)
           |
           v
  02_spearman_decorrelation.py
  For each model: highly correlated feature pairs (|rho| >= 0.7)
  are identified with Spearman correlation. Within each correlated
  group, only the feature ranked highest by that model's SHAP is
  kept — the rest are redundant and dropped.
  --> results/decorr_features_{model}.json   (one per model)
           |
           v
  03_forward_stepwise.py
  For each model: features are added one at a time. At each step,
  every remaining candidate is trialled by 5-fold CV and the one
  that maximises CV F1 is selected. Stops at 40 features or after
  5 consecutive steps with no improvement in best-yet CV F1.
  --> results/stepwise_results.json   (full history per model)
           |
           v
  04_kneedle_final_shap.py
  Kneedle algorithm finds the elbow of each F1 vs. n_features curve.
  The best model at its optimal feature count is tuned with
  RandomizedSearchCV, evaluated on the held-out test set, and
  SHAP values are computed for the final model.
  --> results/final_results.csv
  --> results/final_model_shap.pkl
```

# SOMs Code Pipeline

The Self Organising Maps (SOMs) code is written in MATLAB and combined multiple water related parameters to generate its outputs.
To reproduce the results the following steps are required

1) Select the input variables for the SOM analysis in sw_case.m
2) Run sw_generate_som_input.m to generate csv files used as inputs for the SOM
3) Run sw_generate_som.m to produce output Self organising maps.

This SOMs execution requires the SOM toolbox 2.1 for MATLAB:
https://github.com/ilarinieminen/SOM-Toolbox/blob/master/README



