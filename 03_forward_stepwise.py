"""
Step 3: Forward stepwise feature selection — greedily adds the feature that
maximises 5-fold CV F1 at each step; stops at 40 features or after 5 consecutive
steps with no improvement.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR  = Path('results')
RANDOM_STATE = 42
CV_FOLDS     = 5
MAX_FEATURES = 40   # stop after this many features regardless of improvement
PATIENCE     = 5    # stop early if CV F1 has not improved for this many steps

MODEL_NAMES = ['RandomForest', 'GradientBoosting', 'XGBoost']

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading imputed training data...")
X_train = pd.read_parquet(RESULTS_DIR / 'X_train.parquet')
y_train = pd.read_csv(RESULTS_DIR / 'y_train.csv', index_col=0).squeeze()
print(f"Train: {len(X_train)} samples  |  Positive class: {y_train.sum()} ({y_train.mean()*100:.1f}%)")

weights      = compute_sample_weight(class_weight='balanced', y=y_train)
weight_ratio = float(max(weights) / min(weights))

# =============================================================================
# MODEL FACTORY
# =============================================================================

def make_model(model_name, weight_ratio):
    """Return a fresh model instance. Re-called for each CV fold so that
    no state is shared between folds."""
    if model_name == 'RandomForest':
        return RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=RANDOM_STATE, n_jobs=-1
        )
    elif model_name == 'GradientBoosting':
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
            random_state=RANDOM_STATE
        )
    elif model_name == 'XGBoost':
        return XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=weight_ratio,
            random_state=RANDOM_STATE, tree_method='hist', n_jobs=-1
        )

# =============================================================================
# CROSS-VALIDATED F1 (with per-fold optimal threshold)
# =============================================================================

def cv_f1(model_name, features, weight_ratio):
    """Evaluate a feature set with stratified 5-fold CV.

    For each fold, the classification threshold is set to the value on the
    precision-recall curve that maximises F1 on that fold's validation set.
    The reported score is the mean across folds."""
    skf      = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_f1s = []

    for tr_idx, val_idx in skf.split(X_train[features], y_train):
        X_tr  = X_train[features].iloc[tr_idx]
        X_val = X_train[features].iloc[val_idx]
        y_tr  = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        model = make_model(model_name, weight_ratio)
        if model_name == 'GradientBoosting':
            model.fit(X_tr, y_tr, sample_weight=weights[tr_idx])
        else:
            model.fit(X_tr, y_tr)

        probs    = model.predict_proba(X_val)[:, 1]
        prec_arr, rec_arr, _ = precision_recall_curve(y_val, probs)
        f1_arr   = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-10)
        fold_f1s.append(float(np.max(f1_arr)))

    return float(np.mean(fold_f1s)), float(np.std(fold_f1s))

# =============================================================================
# FORWARD STEPWISE SELECTION
# =============================================================================

all_results = {}

for model_name in MODEL_NAMES:
    print(f"\n{'='*65}")
    print(f"  {model_name}")
    print(f"{'='*65}")

    with open(RESULTS_DIR / f'decorr_features_{model_name}.json') as f:
        candidates = json.load(f)
    print(f"  Candidates after decorrelation: {len(candidates)}")
    print(f"  Stopping: max {MAX_FEATURES} features or {PATIENCE} steps without improvement\n")

    selected       = []
    history        = []
    best_f1_so_far = 0.0
    no_improve     = 0

    for step in range(1, MAX_FEATURES + 1):
        remaining = [f for f in candidates if f not in selected]
        if not remaining:
            print("  No remaining candidates — stopping.")
            break

        # Trial every remaining candidate and pick the best
        best_cand_f1, best_cand_std, best_cand = -1.0, 0.0, None
        for candidate in remaining:
            f1_mean, f1_std = cv_f1(model_name, selected + [candidate], weight_ratio)
            if f1_mean > best_cand_f1:
                best_cand_f1, best_cand_std, best_cand = f1_mean, f1_std, candidate

        # Always add the best candidate for this step
        selected = selected + [best_cand]

        # Track whether the overall best CV F1 has improved
        if best_cand_f1 > best_f1_so_far:
            best_f1_so_far = best_cand_f1
            no_improve     = 0
            marker = '  <-- new best'
        else:
            no_improve += 1
            marker = f'  (no improvement {no_improve}/{PATIENCE})'

        history.append({
            'step':              step,
            'n_features':        len(selected),
            'feature_added':     best_cand,
            'selected_features': selected[:],
            'cv_f1_mean':        best_cand_f1,
            'cv_f1_std':         best_cand_std,
        })

        print(f"  Step {step:2d}: +{best_cand:35s}  F1={best_cand_f1:.4f} +/-{best_cand_std:.4f}{marker}")

        if no_improve >= PATIENCE:
            print(f"\n  Early stopping: no improvement for {PATIENCE} consecutive steps.")
            break

    best_step = max(history, key=lambda h: h['cv_f1_mean'])
    all_results[model_name] = {
        'history':    history,
        'best_step':  best_step,
        'best_cv_f1': best_step['cv_f1_mean'],
        'best_n':     best_step['n_features'],
    }
    print(f"\n  Best: step {best_step['step']} — {best_step['n_features']} features, "
          f"CV F1 = {best_step['cv_f1_mean']:.4f}")
    print(f"  Features: {best_step['selected_features']}")

# =============================================================================
# SAVE AND SUMMARISE
# =============================================================================

with open(RESULTS_DIR / 'stepwise_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*65}")
print("SUMMARY")
print(f"{'='*65}")
for model_name, data in all_results.items():
    print(f"  {model_name:20s}: CV F1 = {data['best_cv_f1']:.4f}  "
          f"at {data['best_n']} features  "
          f"(stopped at step {data['history'][-1]['step']})")

print(f"\nResults saved to: {RESULTS_DIR / 'stepwise_results.json'}")
print("Run 04_kneedle_final_shap.py next.")
