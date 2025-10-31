# XGboostwrapper.py
# run: python XGboostwrapper.py --top_k 38
# Required files: X_train.csv, y_train.csv, X_test.csv

import argparse
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV  # added for grid search

def load_data():
    # Load train/test split from CSV files
    X_train = pd.read_csv("./data/X_train.csv")
    y_train = pd.read_csv("./data/y_train.csv").squeeze()  # convert to Series
    X_test = pd.read_csv("./data/X_test.csv")
    return X_train, y_train, X_test

def xgb_select_features(X_train, y_train, top_k=38):
    # XGBoost model with settings similar to the paper (fixed base params)
    base_model = XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        eta=0.3,           # learning rate (called 'eta' in XGBoost core)
        gamma=0,
        max_depth=6,
        reg_lambda=1,
        eval_metric="logloss",
        random_state=42,
    )

    # Grid search ONLY on these three parameters (multiple values)
    param_grid = {
        "subsample": [0.6, 0.7, 0.8],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "n_estimators": [80,100,120],
    }

    # 5-fold CV with ROC AUC scoring; refit the best model
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    print(f"[GridSearch] Best params: {grid_search.best_params_}")
    print(f"[GridSearch] Best CV AUC: {grid_search.best_score_:.4f}")

    # Train best model on all training data (redundant but explicit)
    model.fit(X_train, y_train)

    # Get feature importance scores
    importances = model.feature_importances_
    # Pick top_k features by importance (descending)
    idx = np.argsort(importances)[::-1][:top_k]
    selected_cols = X_train.columns[idx]
    return selected_cols, importances, idx

def main(top_k):
    X_train, y_train, X_test = load_data()

    # Drop columns that are all-NaN in train
    X_train = X_train.dropna(axis="columns")
    # Align test columns with (cleaned) train columns
    X_test = X_test[X_train.columns]

    selected_cols, importances, idx = xgb_select_features(X_train, y_train, top_k)

    # Save reduced train/test with selected features only
    X_train_sel = X_train[selected_cols].copy()
    X_test_sel = X_test[selected_cols].copy()

    X_train_sel.to_csv("./data/X_train_xgbsel.csv", index=False)
    X_test_sel.to_csv("./data/X_test_xgbsel.csv", index=False)

    # Save selected feature names with importance for inspection
    pd.DataFrame({
        "feature": X_train.columns[idx],
        "importance": importances[idx]
    }).to_csv("./data/xgb_feature_importance.csv", index=False)

    print(f"[XGBoost FS] top {top_k} features saved to X_train_xgbsel.csv / X_test_xgbsel.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=38, help="number of top features to keep")
    args = parser.parse_args()
    main(args.top_k)
