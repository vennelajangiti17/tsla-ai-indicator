"""
=============================================================
Phase 2 | Step 5: Train XGBoost Model with Walk-Forward Validation
=============================================================
Major Improvements:
  1. 5-Fold Walk-Forward Cross Validation:
     - Rolling-origin expanding window protocol.
     - Saves full metrics + 95% Confidence Intervals to results/.
  2. Realistic Trader Confidence Threshold (0.55 / 0.45):
     - Avoids overtrading by abstaining (HOLD) on choppy 50/50 candles.
     - Coverage tuned to ~55-60%, providing genuine selective value.
  3. Honest Statistical Reporting:
     - Reports 95% Confidence Intervals: p +/- 1.96 * sqrt(p(1-p)/N).
  4. Metadata Consistency:
     - Stores lookahead_candles = 2 (10 min), threshold = 0.55 / 0.45.
=============================================================
"""

import os, sys, json, joblib, warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

warnings.filterwarnings('ignore')

CSV_PATH    = "data/tsla_features.csv"
MODEL_PATH  = "models/tsla_xgb_model.pkl"
RESULTS_DIR = "results"
os.makedirs("models", exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_FOLDS   = 5
CONF_HIGH = 0.55
CONF_LOW  = 0.45

DROP_COLS = [
    'Open','High','Low','Close','Volume',
    'future_close','future_time','future_pct_change','target'
]

def load_and_split():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: '{CSV_PATH}' not found. Run step4 first.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df['target'].astype(int)

    n = len(df)
    train_end = int(n * 0.70)
    val_end   = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val     = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test   = X.iloc[val_end:], y.iloc[val_end:]

    print(f"  Loaded {len(df):,} regular session rows, {len(feature_cols)} stationary features")
    print(f"  TRAIN : {len(X_train):,} rows ({str(X_train.index.min())[:10]} to {str(X_train.index.max())[:10]})")
    print(f"  VAL   : {len(X_val):,} rows ({str(X_val.index.min())[:10]} to {str(X_val.index.max())[:10]})")
    print(f"  TEST  : {len(X_test):,} rows ({str(X_test.index.min())[:10]} to {str(X_test.index.max())[:10]}) [LOCKED]")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

def run_walk_forward(X_train, y_train):
    print(f"\n" + "=" * 58)
    print(f"  WALK-FORWARD VALIDATION ({N_FOLDS} Folds)")
    print("=" * 58)
    print("  Rolling-origin evaluation simulating live sequential trading...\n")

    n = len(X_train)
    fold_size = n // (N_FOLDS + 1)
    fold_records = []

    for fold in range(1, N_FOLDS + 1):
        train_end = fold * fold_size
        val_start = train_end
        val_end   = min(val_start + fold_size, n)

        X_f_train = X_train.iloc[:train_end]
        y_f_train = y_train.iloc[:train_end]
        X_f_val   = X_train.iloc[val_start:val_end]
        y_f_val   = y_train.iloc[val_start:val_end]

        fold_model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=15,
            gamma=0.5,
            reg_alpha=0.5,
            reg_lambda=2.0,
            eval_metric='logloss',
            random_state=42,
            verbosity=0
        )
        fold_model.fit(X_f_train, y_f_train)

        probs = fold_model.predict_proba(X_f_val)[:, 1]
        preds = (probs >= 0.5).astype(int)
        raw_acc = accuracy_score(y_f_val, preds)

        conf_mask = (probs >= CONF_HIGH) | (probs <= CONF_LOW)
        cov = conf_mask.mean()
        if conf_mask.sum() > 0:
            conf_preds = (probs[conf_mask] >= 0.5).astype(int)
            conf_acc = accuracy_score(y_f_val[conf_mask], conf_preds)
            up_mask = probs >= CONF_HIGH
            conf_prec_up = precision_score(y_f_val[up_mask], (probs[up_mask] >= 0.5).astype(int), zero_division=0) if up_mask.sum() > 0 else 0.0
            ci_95 = 1.96 * np.sqrt(conf_acc * (1 - conf_acc) / conf_mask.sum()) * 100
        else:
            conf_acc = conf_prec_up = ci_95 = 0.0

        fold_records.append({
            'fold': fold,
            'train_samples': len(X_f_train),
            'test_samples': len(X_f_val),
            'raw_accuracy': round(raw_acc * 100, 2),
            'confident_accuracy': round(conf_acc * 100, 2),
            'precision_up': round(conf_prec_up * 100, 2),
            'signal_coverage': round(cov * 100, 1),
            'ci_95': round(ci_95, 2)
        })

        bar = "█" * int(raw_acc * 20) + "░" * (20 - int(raw_acc * 20))
        print(f"  Fold {fold}: [{bar}] Raw={raw_acc*100:.1f}%  Confident={conf_acc*100:.1f}% (+/-{ci_95:.1f}%)  UP Prec={conf_prec_up*100:.1f}%  Cov={cov*100:.0f}%")

    wf_df = pd.DataFrame(fold_records)
    wf_df.to_csv(os.path.join(RESULTS_DIR, "walk_forward_metrics.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "walk_forward_metrics.json"), "w") as f:
        json.dump(fold_records, f, indent=2)

    avg_raw = wf_df['raw_accuracy'].mean()
    avg_conf = wf_df['confident_accuracy'].mean()
    print(f"\n  Average Walk-Forward Accuracy : {avg_raw:.2f}% (Raw) | {avg_conf:.2f}% (Confident)")
    print(f"  Saved walk-forward metrics to {RESULTS_DIR}/walk_forward_metrics.csv")
    return avg_raw, avg_conf

def train_final_model(X_train, y_train):
    print(f"\n" + "=" * 58)
    print("  TRAINING FINAL MODEL on full training set...")
    print("=" * 58)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=15,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric='logloss',
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    print("  Trained 300 trees ✓")
    return model

def evaluate(model, X, y, label="Validation"):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    raw_acc  = accuracy_score(y, preds)
    raw_prec = precision_score(y, preds, zero_division=0)
    raw_rec  = recall_score(y, preds, zero_division=0)
    raw_f1   = f1_score(y, preds, zero_division=0)
    cm       = confusion_matrix(y, preds)

    conf_mask = (probs >= CONF_HIGH) | (probs <= CONF_LOW)
    cov = conf_mask.mean()
    n_conf = conf_mask.sum()

    if n_conf > 0:
        conf_preds = (probs[conf_mask] >= 0.5).astype(int)
        conf_acc  = accuracy_score(y[conf_mask], conf_preds)
        ci_95     = 1.96 * np.sqrt(conf_acc * (1 - conf_acc) / n_conf) * 100
        up_mask   = probs >= CONF_HIGH
        conf_prec_up = precision_score(y[up_mask], (probs[up_mask] >= 0.5).astype(int), zero_division=0) if up_mask.sum() > 0 else 0.0
        down_mask = probs <= CONF_LOW
        conf_prec_down = precision_score(y[down_mask] == 0, (probs[down_mask] < 0.5), zero_division=0) if down_mask.sum() > 0 else 0.0
    else:
        conf_acc = conf_prec_up = conf_prec_down = ci_95 = 0.0

    print(f"\n" + "-" * 58)
    print(f"  {label.upper()} RESULTS (Threshold: {CONF_HIGH}/{CONF_LOW})")
    print("-" * 58)
    print(f"  Raw Accuracy        : {raw_acc*100:.1f}%")
    print(f"  Raw Precision (UP)  : {raw_prec*100:.1f}%")
    print(f"  Raw F1 Score        : {raw_f1*100:.1f}%")
    print(f"  Confusion Matrix    : TN={cm[0][0]}, FP={cm[0][1]} | FN={cm[1][0]}, TP={cm[1][1]}")
    print(f"  Confident Accuracy  : {conf_acc*100:.1f}%  (+/- {ci_95:.1f}% at 95% CI)")
    print(f"  Confident Prec (UP) : {conf_prec_up*100:.1f}%")
    print(f"  Confident Prec (DN) : {conf_prec_down*100:.1f}%")
    print(f"  Signal Coverage     : {cov*100:.1f}%  (Abstains on {100-cov*100:.1f}% of candles)")

    return {
        'raw_accuracy': raw_acc,
        'conf_accuracy': conf_acc,
        'ci_95': ci_95,
        'conf_prec_up': conf_prec_up,
        'conf_prec_down': conf_prec_down,
        'coverage': cov,
        'cm': cm.tolist()
    }

def save_model(model, feature_cols):
    package = {
        'model': model,
        'feature_cols': feature_cols,
        'confidence_high': CONF_HIGH,
        'confidence_low': CONF_LOW,
        'lookahead_candles': 2,
        'timeframe_minutes': 10
    }
    joblib.dump(package, MODEL_PATH)
    print(f"\n  Model Package saved to: {MODEL_PATH}")
    print(f"  Metadata: Lookahead=2 candles (10 min), Threshold: UP>={CONF_HIGH}, DOWN<={CONF_LOW}")

if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_and_split()
    run_walk_forward(X_train, y_train)
    model = train_final_model(X_train, y_train)
    val_res = evaluate(model, X_val, y_val, label="Validation Set")
    test_res = evaluate(model, X_test, y_test, label="Held-Out Test Set (Single Pass)")
    save_model(model, feature_cols)
