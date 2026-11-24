"""
Step 4: Applies Kneedle to each model's stepwise curve to select the optimal
feature count, tunes the best model, sets the decision threshold from out-of-fold
predictions on the training set, evaluates once on the held-out test set, and
computes final SHAP values.

Model choice, hyperparameters and the decision threshold are all determined from
the training set; the test set is used only for the final evaluation.
"""

import json
import pickle
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from kneed import KneeLocator
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR  = Path('results')
RANDOM_STATE = 42
CV_FOLDS     = 5

# Hyperparameter search spaces (60 random draws each)
PARAM_GRIDS = {
    'RandomForest': {
        'n_estimators':      [200, 400, 600, 800],
        'max_depth':         [10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf':  [1, 2, 4],
        'max_features':      [0.3, 0.5, 0.7, 'sqrt', 'log2'],
        'class_weight':      ['balanced', 'balanced_subsample'],
    },
    'GradientBoosting': {
        'n_estimators':  [100, 200, 400, 600],
        'max_depth':     [3, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
        'subsample':     [0.6, 0.7, 0.8, 0.9, 1.0],
        'max_features':  [0.5, 0.7, None, 'sqrt'],
    },
    'XGBoost': {
        'n_estimators':     [200, 400, 600, 800],
        'max_depth':        [3, 5, 6, 8, 10],
        'learning_rate':    [0.01, 0.05, 0.1, 0.15],
        'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
        'gamma':            [0, 0.05, 0.1, 0.3, 0.5],
        'reg_alpha':        [0, 0.1, 0.5, 1],
        'reg_lambda':       [0.5, 1, 1.5, 2],
    },
}

# =============================================================================
# LOAD DATA AND STEPWISE RESULTS
# =============================================================================

print("Loading data and stepwise results...")
X_train = pd.read_parquet(RESULTS_DIR / 'X_train.parquet')
X_test  = pd.read_parquet(RESULTS_DIR / 'X_test.parquet')
y_train = pd.read_csv(RESULTS_DIR / 'y_train.csv', index_col=0).squeeze()
y_test  = pd.read_csv(RESULTS_DIR / 'y_test.csv',  index_col=0).squeeze()

with open(RESULTS_DIR / 'stepwise_results.json') as f:
    stepwise_results = json.load(f)

weights      = compute_sample_weight(class_weight='balanced', y=y_train)
weight_ratio = float(max(weights) / min(weights))

# =============================================================================
# KNEEDLE: FIND OPTIMAL FEATURE COUNT FOR EACH MODEL
# =============================================================================

print("\nApplying Kneedle algorithm to each model's stepwise F1 curve...")

kneedle_summary = {}

for model_name, data in stepwise_results.items():
    history = data['history']
    steps   = [h['n_features'] for h in history]
    scores  = [h['cv_f1_mean'] for h in history]

    # Try decreasing sensitivity (S) until a knee is found.
    # S=1.0 is the default; lower values accept subtler elbows.
    knee_n = None
    for S in [1.0, 0.5, 0.25, 0.1]:
        kl = KneeLocator(steps, scores, curve='concave',
                         direction='increasing', S=S, online=True)
        if kl.knee is not None:
            knee_n = kl.knee
            print(f"  {model_name}: knee at {knee_n} features (S={S}), "
                  f"CV F1 = {scores[steps.index(knee_n)]:.4f}")
            break

    if knee_n is None:
        knee_n = steps[int(np.argmax(scores))]
        print(f"  {model_name}: no knee detected — using peak at {knee_n} features, "
              f"CV F1 = {max(scores):.4f}")

    knee_step = next(h for h in history if h['n_features'] == knee_n)
    kneedle_summary[model_name] = {
        'knee_n':   knee_n,
        'knee_f1':  knee_step['cv_f1_mean'],
        'features': knee_step['selected_features'],
    }

# =============================================================================
# SELECT BEST MODEL AT ITS KNEEDLE-OPTIMAL FEATURE COUNT
# =============================================================================

best_model_name = max(kneedle_summary, key=lambda m: kneedle_summary[m]['knee_f1'])
best_features   = kneedle_summary[best_model_name]['features']

print(f"\n{'='*65}")
print(f"Best model:    {best_model_name}")
print(f"Feature count: {len(best_features)}  (kneedle-optimal)")
print(f"CV F1:         {kneedle_summary[best_model_name]['knee_f1']:.4f}")
print(f"Features:      {best_features}")
print(f"{'='*65}")

# =============================================================================
# HYPERPARAMETER TUNING
# =============================================================================

print(f"\nTuning {best_model_name} — 60 random combinations, {CV_FOLDS}-fold CV...")

skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

if best_model_name == 'RandomForest':
    base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
elif best_model_name == 'GradientBoosting':
    base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
else:
    base_model = XGBClassifier(
        objective='binary:logistic', scale_pos_weight=weight_ratio,
        tree_method='hist', n_jobs=-1, random_state=RANDOM_STATE
    )

search = RandomizedSearchCV(
    base_model, PARAM_GRIDS[best_model_name],
    n_iter=60, cv=skf, scoring='average_precision',
    n_jobs=-1, random_state=RANDOM_STATE, verbose=1
)

if best_model_name == 'GradientBoosting':
    search.fit(X_train[best_features], y_train, sample_weight=weights)
else:
    search.fit(X_train[best_features], y_train)

tuned_model = search.best_estimator_
print(f"Best CV average precision: {search.best_score_:.4f}")
print(f"Best hyperparameters:\n  {search.best_params_}")

# =============================================================================
# DECISION THRESHOLD FROM OUT-OF-FOLD TRAINING PREDICTIONS
# =============================================================================
# The probability cut-off is chosen without reference to the test set: the tuned
# model is re-fitted within each CV fold of the training set, the held-out fold is
# predicted, and the threshold that maximises F1 over these out-of-fold
# predictions is taken.

print(f"\nSelecting decision threshold from {CV_FOLDS}-fold out-of-fold training predictions...")

oof_probs = np.zeros(len(y_train))
for tr_idx, val_idx in skf.split(X_train[best_features], y_train):
    fold_model = clone(tuned_model)
    X_tr, y_tr = X_train[best_features].iloc[tr_idx], y_train.iloc[tr_idx]
    if best_model_name == 'GradientBoosting':
        fold_model.fit(X_tr, y_tr, sample_weight=weights[tr_idx])
    else:
        fold_model.fit(X_tr, y_tr)
    oof_probs[val_idx] = fold_model.predict_proba(X_train[best_features].iloc[val_idx])[:, 1]

prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_train, oof_probs)
f1_arr     = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-10)
best_idx   = int(np.argmax(f1_arr[:-1]))   # last PR point has no threshold
opt_thresh = float(thresh_arr[best_idx])
oof_f1     = float(f1_arr[best_idx])
print(f"Threshold: {opt_thresh:.4f}  (out-of-fold training F1 = {oof_f1:.4f})")

# =============================================================================
# TEST SET EVALUATION  (single evaluation of the fixed model and threshold)
# =============================================================================

probs  = tuned_model.predict_proba(X_test[best_features])[:, 1]
y_pred = (probs >= opt_thresh).astype(int)

prec = float(precision_score(y_test, y_pred))
rec  = float(recall_score(y_test, y_pred))
f1   = float(f1_score(y_test, y_pred))

print(f"\n--- Test Set Results ---")
print(f"Model:          {best_model_name} ({len(best_features)} features)")
print(f"Threshold:      {opt_thresh:.4f}  (F1-optimal on out-of-fold training predictions)")
print(f"Precision:      {prec:.4f}")
print(f"Recall:         {rec:.4f}")
print(f"F1:             {f1:.4f}")

pd.DataFrame([{
    'model':       best_model_name,
    'n_features':  len(best_features),
    'features':    ', '.join(best_features),
    'threshold':   opt_thresh,
    'oof_train_f1': oof_f1,
    'precision':   prec,
    'recall':      rec,
    'f1':          f1,
    'best_params': str(search.best_params_),
}]).to_csv(RESULTS_DIR / 'final_results.csv', index=False)

# =============================================================================
# FINAL SHAP ANALYSIS
# =============================================================================

print(f"\nComputing SHAP values for the final tuned {best_model_name}...")

try:
    explainer   = shap.Explainer(tuned_model, X_train[best_features])
    shap_values = explainer(X_test[best_features], check_additivity=False)
except Exception as err:
    # Some shap/xgboost version combinations reject the background-data explainer;
    # the path-dependent tree explainer gives the same attributions up to the
    # treatment of feature dependence.
    print(f"  Background-data explainer unavailable ({type(err).__name__}); using TreeExplainer")
    shap_values = shap.TreeExplainer(tuned_model)(X_test[best_features], check_additivity=False)

# Keep the positive class where the explainer returns one column per class
# (e.g. RandomForest: values of shape (rows, features, 2)).
if np.ndim(shap_values.values) == 3:
    shap_values = shap_values[:, :, 1]

with open(RESULTS_DIR / 'final_model_shap.pkl', 'wb') as f:
    pickle.dump({
        'shap':              shap_values,
        'X_test':            X_test[best_features],
        'features':          best_features,
        'model':             best_model_name,
        'best_params':       search.best_params_,
        'optimal_threshold': opt_thresh,
        'metrics': {'precision': prec, 'recall': rec, 'f1': f1},
    }, f)

print(f"\nOutputs saved to {RESULTS_DIR}/")
print("  final_results.csv    — metrics and selected features")
print("  final_model_shap.pkl — SHAP values and model metadata")
