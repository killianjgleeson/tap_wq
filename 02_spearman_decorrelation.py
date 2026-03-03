"""
Step 2: Per-model Spearman decorrelation — removes redundant correlated features,
keeping the highest SHAP-ranked member from each correlated group.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR    = Path('results')
CORR_THRESHOLD = 0.7   # pairs with |Spearman rho| >= this are treated as redundant
MODEL_NAMES    = ['RandomForest', 'GradientBoosting', 'XGBoost']

# =============================================================================
# LOAD TRAINING DATA
# =============================================================================

print("Loading imputed training data...")
X_train = pd.read_parquet(RESULTS_DIR / 'X_train.parquet')
all_features = X_train.columns.tolist()
print(f"Features available: {len(all_features)}")

# =============================================================================
# UNION-FIND HELPERS
# =============================================================================
# Union-find (disjoint set) efficiently groups correlated features into
# clusters so we can keep one representative from each cluster.

def find_root(parent, x):
    """Path-compressed root finder — flattens the tree for speed."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]   # path compression
        x = parent[x]
    return x


def union_groups(parent, a, b):
    """Merge the clusters containing a and b."""
    ra, rb = find_root(parent, a), find_root(parent, b)
    if ra != rb:
        parent[rb] = ra


# =============================================================================
# PER-MODEL DECORRELATION
# =============================================================================

for model_name in MODEL_NAMES:
    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"{'='*60}")

    # Load this model's SHAP ranking from Step 1
    shap_df    = pd.read_csv(RESULTS_DIR / f'shap_{model_name}.csv')
    shap_order = {row['feature']: i for i, row in shap_df.iterrows()}
    # Keep only features that exist in the imputed data
    candidates = [f for f in shap_df['feature'] if f in all_features]
    print(f"  Candidate features: {len(candidates)}")

    # -------------------------------------------------------------------------
    # Compute Spearman correlation matrix
    # -------------------------------------------------------------------------
    corr_matrix = X_train[candidates].corr(method='spearman')
    corr_matrix.to_csv(RESULTS_DIR / f'corr_matrix_{model_name}.csv')

    # Find all pairs above the threshold
    corr_pairs = []
    for i, f1 in enumerate(candidates):
        for f2 in candidates[i+1:]:
            rho = corr_matrix.loc[f1, f2]
            if abs(rho) >= CORR_THRESHOLD:
                corr_pairs.append({'feature_1': f1, 'feature_2': f2, 'spearman_rho': rho})

    pd.DataFrame(corr_pairs).to_csv(RESULTS_DIR / f'corr_pairs_{model_name}.csv', index=False)
    print(f"  Correlated pairs (|rho| >= {CORR_THRESHOLD}): {len(corr_pairs)}")

    if corr_pairs:
        print(f"  {'Feature 1':30s}  {'Feature 2':30s}  rho")
        for p in sorted(corr_pairs, key=lambda x: -abs(x['spearman_rho'])):
            print(f"  {p['feature_1']:30s}  {p['feature_2']:30s}  {p['spearman_rho']:+.3f}")

    # -------------------------------------------------------------------------
    # Group correlated features with union-find and keep the best per group
    # -------------------------------------------------------------------------
    parent = {f: f for f in candidates}
    for p in corr_pairs:
        union_groups(parent, p['feature_1'], p['feature_2'])

    # Collect groups (each key is a root, value is the list of members)
    groups = {}
    for f in candidates:
        root = find_root(parent, f)
        groups.setdefault(root, []).append(f)

    # Within each correlated group, keep the feature with the best SHAP rank
    # (lowest index in shap_order = highest importance for this model).
    kept    = []
    dropped = []
    for members in groups.values():
        best = min(members, key=lambda f: shap_order.get(f, 9999))
        kept.append(best)
        dropped.extend([f for f in members if f != best])

    if dropped:
        print(f"\n  Groups where a feature was dropped:")
        for members in groups.values():
            if len(members) > 1:
                members_ranked = sorted(members, key=lambda f: shap_order.get(f, 9999))
                print(f"    Keep : {members_ranked[0]}  (SHAP rank {shap_order.get(members_ranked[0], '?')+1})")
                for f in members_ranked[1:]:
                    print(f"    Drop : {f}  (SHAP rank {shap_order.get(f, '?')+1})")

    # Restore SHAP rank order so the stepwise tries the most promising features first
    kept_ordered = sorted(kept, key=lambda f: shap_order.get(f, 9999))
    print(f"\n  Features after decorrelation: {len(kept_ordered)} / {len(candidates)}")

    out_path = RESULTS_DIR / f'decorr_features_{model_name}.json'
    with open(out_path, 'w') as f:
        json.dump(kept_ordered, f, indent=2)
    print(f"  Saved: {out_path.name}")

print("\nStep 2 complete. Run 03_forward_stepwise.py next.")
