"""
Step 1: Per-model SHAP ranking — trains each model on all features and computes
SHAP values to produce a model-specific feature importance ordering.
Also performs the train/test split and MICE imputation shared by all subsequent steps.

The ranking is computed on training rows only, so the held-out test set plays no
part in feature selection and is first used in Step 4 for the final evaluation.
"""

import pandas as pd
import numpy as np
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required before importing IterativeImputer)
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# =============================================================================
# CONFIGURATION  —  update DATA_PATH before running
# =============================================================================

DATA_PATH  = 'path/to/your/data.csv'   # <-- set this to your dataset
RESULTS_DIR = Path('results')

TARGET    = 'CL2FREE_1'   # free chlorine residual at the tap (mg/L)
THRESHOLD = 0.1           # below this = low chlorine event (binary label = 1)
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Number of training rows on which SHAP values are computed for the ranking.
# TreeSHAP cost scales with the number of rows explained; a random sample of the
# training set gives a stable ranking. Set to None to use every training row.
SHAP_SAMPLE_ROWS = 20000

# Domain-knowledge exclusions applied before any modelling.
# - longitude/latitude: geographic coordinates that would cause location overfitting
# - mn: raw month integer, redundant with the sinusoidal month_sin encoding
# - CC: customer contact counts (outcome, not predictor)
# - VolumeperWTW: WTW-specific rather than DMA-aggregate; meaningless across DMAs
#   served by different numbers of WTWs
# - HP_Dead_end flags: hydraulic model artefacts excluded by domain agreement
EXCLUDED_FEATURES = ['longitude', 'latitude', 'mn']
EXCLUDED_PATTERNS = ['CC', 'VolumeperWTW', 'HP_Dead_end', 'HP_Weighted_Dead_end']

RESULTS_DIR.mkdir(exist_ok=True)

# =============================================================================
# LOAD AND CLEAN DATA
# =============================================================================

print("Loading data...")
df = pd.read_csv(DATA_PATH, header=0, index_col=0)

df = df.dropna(subset=[TARGET]).drop_duplicates()
y  = (df[TARGET] < THRESHOLD).astype(int)
print(f"Samples: {len(df)}  |  Low Cl events: {y.sum()} ({y.mean()*100:.1f}%)")

# Build the feature matrix:
#   - numeric columns only (drops text identifiers automatically)
#   - remove metadata not useful as predictors
#   - remove tap-level measurements (_1 suffix) — these are the response
#     variables for other WQ outcomes and would cause data leakage here
#   - apply domain exclusions
X = df.drop(columns=[TARGET]).select_dtypes(include=[np.number])

metadata  = ['SampleNo', 'GroupNo', 'yr', 'Time', 'YWREF',
             'DIAMETER', 'Weight', 'DMAWeight']
tap_cols  = [c for c in X.columns if c.endswith('_1')]
excl_cols = [c for c in X.columns
             if c in EXCLUDED_FEATURES or any(p in c for p in EXCLUDED_PATTERNS)]

X = X.drop(columns=metadata + tap_cols + excl_cols, errors='ignore')
print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

# =============================================================================
# TRAIN/TEST SPLIT AND MICE IMPUTATION
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print("\nRunning MICE imputation (this may take a few minutes)...")
imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
cols    = X.columns.tolist()
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=cols, index=X_train.index)
X_test  = pd.DataFrame(imputer.transform(X_test),      columns=cols, index=X_test.index)

X_train.to_parquet(RESULTS_DIR / 'X_train.parquet')
X_test.to_parquet(RESULTS_DIR  / 'X_test.parquet')
y_train.to_csv(RESULTS_DIR / 'y_train.csv')
y_test.to_csv(RESULTS_DIR  / 'y_test.csv')
print(f"Imputed data saved  ({len(X_train)} train, {len(X_test)} test samples)")

# Class imbalance: weight ratio used by XGBoost's scale_pos_weight;
# sample weights used by GradientBoosting during fit.
weights      = compute_sample_weight(class_weight='balanced', y=y_train)
weight_ratio = float(max(weights) / min(weights))
print(f"Class weight ratio (majority:minority): {weight_ratio:.1f}:1")

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_split=5,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
        random_state=RANDOM_STATE
    ),
    'XGBoost': XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=weight_ratio,
        random_state=RANDOM_STATE, tree_method='hist', n_jobs=-1
    ),
}

# =============================================================================
# PER-MODEL SHAP RANKING
# =============================================================================

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    print(f"  Fitting on {len(X_train)} training samples...")
    if model_name == 'GradientBoosting':
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        model.fit(X_train, y_train)

    # SHAP values are computed on training rows only (never on the test set).
    if SHAP_SAMPLE_ROWS is not None and len(X_train) > SHAP_SAMPLE_ROWS:
        X_shap = X_train.sample(n=SHAP_SAMPLE_ROWS, random_state=RANDOM_STATE)
    else:
        X_shap = X_train

    print(f"  Computing SHAP values on {len(X_shap)} training samples...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_shap)

    # Select the positive class. Depending on the model and the shap version the
    # output is a list [class_0, class_1], a 3-D array (rows, features, classes),
    # or a single 2-D array that already refers to the positive class.
    if isinstance(shap_vals, list):
        shap_class1 = shap_vals[1]
    elif np.ndim(shap_vals) == 3:
        shap_class1 = shap_vals[:, :, 1]
    else:
        shap_class1 = shap_vals

    importance = pd.DataFrame({
        'feature':       cols,
        'mean_abs_shap': np.abs(shap_class1).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    out_path = RESULTS_DIR / f'shap_{model_name}.csv'
    importance.to_csv(out_path, index=False)

    print(f"  Top 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {int(row.name)+1:2d}. {row['feature']:35s}  {row['mean_abs_shap']:.5f}")
    print(f"  Saved: {out_path.name}")

print("\nStep 1 complete. Run 02_spearman_decorrelation.py next.")
