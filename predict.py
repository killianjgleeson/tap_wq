"""
Makes predictions on new data with one of the trained final models.

Usage: python predict.py chlorine my_data.csv
"""

import sys
import json
import joblib
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier

if len(sys.argv) != 3:
    sys.exit(__doc__)
target, input_csv = sys.argv[1], sys.argv[2]

# model, input columns and decision threshold
folder = Path(__file__).parent / 'models' / target
info = json.loads((folder / 'model_info.json').read_text())
features = [f['column'] for f in info['features']]

if info['algorithm'] == 'XGBoost':
    model = XGBClassifier()
    model.load_model(folder / info['model_file'])
else:
    model = joblib.load(folder / info['model_file'])

# rows with a missing value are not scored
data = pd.read_csv(input_csv)
rows = data[features].notna().all(axis=1)

data.loc[rows, 'probability'] = model.predict_proba(data.loc[rows, features])[:, 1]
data['predicted_event'] = (data['probability'] >= info['decision_threshold']).astype('Int64').where(rows)

data.to_csv('predictions.csv', index=False)
print(f"{rows.sum()} of {len(data)} rows scored, saved to predictions.csv")
