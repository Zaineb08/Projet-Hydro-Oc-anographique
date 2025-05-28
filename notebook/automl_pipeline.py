#!/usr/bin/env python3
"""
automl_pipeline.py

Run an AutoML pipeline with PyCaret to find and evaluate the best model for your 'target'.

Usage:
  1. Install dependencies:
     pip install pycaret[full] pandas numpy
  2. Place 'processed_data.csv' in the same directory as this script.
  3. Execute:
     python automl_pipeline.py
"""

import pandas as pd
import numpy as np
from pycaret.classification import setup, compare_models, tune_model, evaluate_model, save_model

# 1. Load data
df = pd.read_csv('processed_data.csv')

# 2. Feature engineering
df['wind_speed']    = np.sqrt(df['u10']**2 + df['v10']**2)
df['current_speed'] = np.sqrt(df['u_curr']**2 + df['v_curr']**2)

# 3. AutoML setup
clf = setup(
    data=df,
    target='target',
    train_size=0.8,
    session_id=42,
    silent=True,
    verbose=False
)

# 4. Compare models and tune the best
best = compare_models(sort='AUC')
tuned = tune_model(best, optimize='AUC')

# 5. Evaluate and save
evaluate_model(tuned)
save_model(tuned, 'best_model_pipeline')
print("\nSaved the tuned model as 'best_model_pipeline.pkl'")
