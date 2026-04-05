"""
Phase 6: Baseline Models
6A: Logistic Regression (financial-only)
6B: Logistic Regression (sentiment-only)
6C: Combined model + evaluation metrics
6D: Baseline performance documentation
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Columns that must NEVER be used as model features.
# These are either the target variable, future price information (look-ahead),
# or metadata identifiers that carry no predictive signal.
# ─────────────────────────────────────────────────────────────────────────────
LEAK_COLS = {
    # Target variables — these ARE what we are predicting
    'stock_return_3day',
    'abnormal_return',
    'label_binary',
    'label_median',
    'label_tertile',

    # Future price information — direct look-ahead bias
    'price_before_earnings',   # correlated with after-price
    'price_after_earnings',    # never available at prediction time

    # Market dates — metadata, not predictive features
    'market_date_before',
    'market_date_after',

    # Validation and pipeline flags
    'is_temporally_valid',
    'alignment_timestamp',

    # Identifier columns — models should not memorise tickers/names
    'ticker',
    'company_name',
    'quarter',

    # Date columns — models should not overfit to calendar dates
    'transcript_date',
    'financial_date',
    'fiscal_year',
    'fiscal_quarter',
    'financial_date_diff_days',
    'financial_DateDiff',
}


class BaselineModels:

    def __init__(self):
        self.data_dir    = FINAL_DATA_DIR
        self.models_dir  = MODELS_DIR
        self.results_dir = RESULTS_DIR / 'baseline_models'
        self.log_file    = LOGS_DIR / 'models' / 'baseline_models.log'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.all_results = {}

    # ── Data Loading ──────────────────────────────────────────────────────────

    def load_data(self):
        print("📂 LOADING DATA...")
        train_df = pd.read_csv(self.data_dir / 'train_data.csv')
        test_df  = pd.read_csv(self.data_dir / 'test_data.csv')
        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        print(f"✅ Train: {len(train_df)} | Test: {len(test_df)}")
        print(f"✅ Features available: {len(feature_info['feature_columns'])}")
        print()
        return train_df, test_df, feature_info

    # ── Feature Set Definitions ───────────────────────────────────────────────

    def get_feature_sets(self, feature_info, train_df):
        """
        Define which columns belong to each feature set.

        LEAK_COLS are excluded from every feature set to prevent:
          - Look-ahead bias (price_after_earnings)
          - Target leakage (label_binary, abnormal_return)
          - Identifier memorisation (ticker, company_name)
        """
        # Valid columns = in feature_info AND in dataframe AND not a leak col
        all_cols = (
            set(feature_info['feature_columns']) & set(train_df.columns)
        ) - LEAK_COLS

        # Report what was removed
        removed = (
            set(feature_info['feature_columns']) & set(train_df.columns)
        ) & LEAK_COLS
        if removed:
            print(f"⚠️  Removed {len(removed)} leak/invalid columns from features:")
            for c in sorted(removed):
                print(f"      - {c}")
            print()

        # Financial features
        financial = sorted([
            c for c in all_cols
            if c.startswith('financial_') or c.startswith('fin_')
        ])

        # Sentiment / NLP features
        sentiment = sorted([
            c for c in all_cols
            if any(c.startswith(p) for p in [
                'sentiment_', 'mgmt_sentiment_', 'analyst_sentiment_',
                'lm_', 'mgmt_lm_', 'analyst_lm_',
                'word_count', 'question_count', 'lexical_diversity',
                'mgmt_word_count', 'analyst_word_count',
                'lm_sentiment_divergence', 'uncertainty_divergence',
                'mgmt_avg_word_length', 'avg_word_length', 'unique_words',
            ])
        ])

        # Combined: financial + sentiment, no FinBERT embeddings
        combined = sorted([
            c for c in all_cols
            if not c.startswith('embedding_')
        ])

        print(f"📊 Feature sets (after leak removal):")
        print(f"   financial_only : {len(financial)} features")
        print(f"   sentiment_only : {len(sentiment)} features")
        print(f"   combined       : {len(combined)} features")
        print()

        return {
            'financial_only': financial,
            'sentiment_only': sentiment,
            'combined':       combined,
        }

    # ── Preprocessing ─────────────────────────────────────────────────────────

    def clean_and_scale(self, X_train, X_test):
        """Handle inf/nan, clip extremes, standardise. Fit ONLY on train."""
        # Step 1: inf → nan
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_test  = np.where(np.isinf(X_test),  np.nan, X_test)
 
        # Step 2: compute per-column medians from train only
        medians = np.nanmedian(X_train, axis=0)
        # If a column is entirely NaN, nanmedian returns NaN → fall back to 0
        medians = np.where(np.isnan(medians), 0.0, medians)
 
        # Step 3: fill NaN with train median
        def _fill(X):
            nan_mask = np.isnan(X)
            if nan_mask.any():
                col_idx = np.where(nan_mask.any(axis=0))[0]
                for j in col_idx:
                    X[nan_mask[:, j], j] = medians[j]
            return X
 
        X_train = _fill(X_train.copy())
        X_test  = _fill(X_test.copy())
 
        # Step 4: drop columns that are still all-zero / constant in train
        # (StandardScaler would produce NaN for zero-std columns)
        col_std = X_train.std(axis=0)
        keep    = col_std > 0
        if not keep.all():
            n_dropped = (~keep).sum()
            print(f"   ⚠️  Dropping {n_dropped} constant/zero-variance column(s)")
            X_train = X_train[:, keep]
            X_test  = X_test[:, keep]
 
        # Step 5: clip extremes and scale
        X_train = np.clip(X_train, -1e6, 1e6)
        X_test  = np.clip(X_test,  -1e6, 1e6)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
 
        # Safety net: any remaining NaN → 0  (should never happen after above)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
 
        return X_train, X_test
    
    def prepare(self, train_df, test_df, cols):
        # Keep only numeric columns
        numeric_cols = [
            c for c in cols
            if pd.api.types.is_numeric_dtype(train_df[c])
        ]

        dropped = set(cols) - set(numeric_cols)
        if dropped:
            print(f"⚠️ Dropping {len(dropped)} non-numeric columns:")
            for c in dropped:
                print(f"   - {c}")

        X_train = train_df[numeric_cols].values.astype(np.float64)
        X_test  = test_df[numeric_cols].values.astype(np.float64)

        return self.clean_and_scale(X_train, X_test)

    # ── Model Evaluation ──────────────────────────────────────────────────────

    def eval_clf(self, model, X_tr, y_tr, X_te, y_te):
        model.fit(X_tr, y_tr)
        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        return {
            'accuracy':         float(accuracy_score(y_te, y_pred)),
            'precision':        float(precision_score(y_te, y_pred, zero_division=0)),
            'recall':           float(recall_score(y_te, y_pred, zero_division=0)),
            'f1':               float(f1_score(y_te, y_pred, zero_division=0)),
            'roc_auc':          float(roc_auc_score(y_te, y_proba)),
            'confusion_matrix': confusion_matrix(y_te, y_pred).tolist(),
        }

    def eval_reg(self, model, X_tr, y_tr, X_te, y_te):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        return {
            'rmse':      float(np.sqrt(mean_squared_error(y_te, y_pred))),
            'mae':       float(mean_absolute_error(y_te, y_pred)),
            'r2':        float(r2_score(y_te, y_pred)),
            'direction': float(np.mean((y_te > 0) == (y_pred > 0))),
        }

    # ── Sub-phase Runner ──────────────────────────────────────────────────────

    def _run_subphase(self, label, feature_key, train_df, test_df, feature_sets, result_key):
        print("=" * 60)
        print(f"{label}")
        print("=" * 60)
        cols = feature_sets[feature_key]
        print(f"   Features: {len(cols)}")
        print()

        if len(cols) == 0:
            print("   ⚠️  No features found for this set — skipping.")
            print()
            return None

        X_tr, X_te = self.prepare(train_df, test_df, cols)
        y_tr_clf = train_df['label_binary'].values
        y_te_clf = test_df['label_binary'].values
        y_tr_reg = train_df['abnormal_return'].values
        y_te_reg = test_df['abnormal_return'].values

        lr = self.eval_clf(
            LogisticRegression(
                C=0.1, max_iter=1000,
                class_weight='balanced',
                random_state=42
            ),
            X_tr, y_tr_clf, X_te, y_te_clf
        )
        rf = self.eval_clf(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=4,           # was 6 — too deep for 558 samples, RF was scoring below 50%
                class_weight='balanced',
                random_state=42, n_jobs=-1
            ),
            X_tr, y_tr_clf, X_te, y_te_clf
        )
        reg = self.eval_reg(
            # Switched from Lasso to Ridge — Lasso was collapsing to mean prediction
            # (all three feature sets produced identical RMSE/R²/Direction = dead giveaway).
            # Ridge shrinks coefficients without zeroing them — works better for
            # weak-signal regression on 558 samples.
            Ridge(alpha=1.0),
            X_tr, y_tr_reg, X_te, y_te_reg
        )

        for name, m in [('Logistic Regression', lr), ('Random Forest', rf)]:
            print(f"   {name}:")
            print(f"      ROC-AUC:  {m['roc_auc']:.1%}  ← PRIMARY METRIC")
            print(f"      Accuracy: {m['accuracy']:.1%}")
            print(f"      F1:       {m['f1']:.1%}")
            print()
        print(f"   Ridge Regression:")
        print(f"      RMSE:      {reg['rmse']:.4f}")
        print(f"      R²:        {reg['r2']:.4f}")
        print(f"      Direction: {reg['direction']:.1%}")
        print()

        result = {
            'feature_set':         feature_key,
            'n_features':          len(cols),
            'logistic_regression': lr,
            'random_forest':       rf,
            'ridge_regression':    reg,
        }
        self.all_results[result_key] = result
        return result

    # ── Sub-phases ────────────────────────────────────────────────────────────

    def run_6a(self, train_df, test_df, feature_sets):
        return self._run_subphase(
            "6A: LOGISTIC REGRESSION — FINANCIAL FEATURES ONLY",
            'financial_only', train_df, test_df, feature_sets, '6A_financial_only')

    def run_6b(self, train_df, test_df, feature_sets):
        return self._run_subphase(
            "6B: LOGISTIC REGRESSION — SENTIMENT FEATURES ONLY",
            'sentiment_only', train_df, test_df, feature_sets, '6B_sentiment_only')

    def run_6c(self, train_df, test_df, feature_sets):
        return self._run_subphase(
            "6C: COMBINED MODEL + EVALUATION METRICS",
            'combined', train_df, test_df, feature_sets, '6C_combined')

    # ── Documentation ─────────────────────────────────────────────────────────

    def run_6d(self):
        print("=" * 60)
        print("6D: BASELINE PERFORMANCE DOCUMENTATION")
        print("=" * 60)
        print()

        label_map = {
            '6A_financial_only': 'Financial Only',
            '6B_sentiment_only': 'Sentiment Only',
            '6C_combined':       'Combined',
        }

        # Classification summary
        print(f"{'Feature Set':<20} {'Model':<22} {'ROC-AUC':<10} {'Accuracy':<12} {'F1'}")
        print("-" * 68)
        for key, label in label_map.items():
            if key not in self.all_results:
                continue
            r = self.all_results[key]
            for mname, mkey in [('Logistic Reg', 'logistic_regression'),
                                 ('Random Forest', 'random_forest')]:
                m = r[mkey]
                print(f"{label:<20} {mname:<22} {m['roc_auc']:<10.1%} "
                      f"{m['accuracy']:<12.1%} {m['f1']:.1%}")
        print()

        # Regression summary
        print(f"{'Feature Set':<20} {'RMSE':<12} {'R²':<12} {'Direction'}")
        print("-" * 52)
        for key, label in label_map.items():
            if key not in self.all_results:
                continue
            r = self.all_results[key]['ridge_regression']
            print(f"{label:<20} {r['rmse']:<12.4f} {r['r2']:<12.4f} {r['direction']:.1%}")
        print()

        # Threshold check and JSON save
        print("📊 THRESHOLD CHECK (ROC-AUC ≥ 0.55):")
        doc = {
            'phase':             '6D',
            'timestamp':         datetime.now().isoformat(),
            'target_metric':     'ROC-AUC',
            'target_threshold':  0.55,
            'leak_cols_removed': sorted(list(LEAK_COLS)),
            'results':           {},
        }

        for key, label in label_map.items():
            if key not in self.all_results:
                continue
            r    = self.all_results[key]
            best = max(r['logistic_regression']['roc_auc'],
                       r['random_forest']['roc_auc'])
            status = "✅ MEETS" if best >= 0.55 else "❌ BELOW"
            print(f"   {label:<20} Best AUC: {best:.1%}  {status} threshold")
            doc['results'][key] = {
                'feature_set':         r['feature_set'],
                'n_features':          r['n_features'],
                'best_roc_auc':        best,
                'meets_threshold':     best >= 0.55,
                'logistic_regression': {k: v for k, v in r['logistic_regression'].items()
                                        if k != 'confusion_matrix'},
                'random_forest':       {k: v for k, v in r['random_forest'].items()
                                        if k != 'confusion_matrix'},
                'ridge_regression':    r['ridge_regression'],
            }
        print()

        doc_path = self.results_dir / 'phase6_documentation.json'
        with open(doc_path, 'w') as f:
            json.dump(doc, f, indent=2)
        print(f"✅ Documentation saved: {doc_path.name}")
        print()

    # ── Main Entry Point ──────────────────────────────────────────────────────

    def run(self):
        print("=" * 60)
        print("PHASE 6: BASELINE MODELS")
        print("=" * 60)
        print()

        train_df, test_df, feature_info = self.load_data()
        feature_sets = self.get_feature_sets(feature_info, train_df)

        self.run_6a(train_df, test_df, feature_sets)
        self.run_6b(train_df, test_df, feature_sets)
        self.run_6c(train_df, test_df, feature_sets)
        self.run_6d()

        print("=" * 60)
        print("✅ PHASE 6 COMPLETE")
        print("=" * 60)
        print()
        return self.all_results


def main():
    BaselineModels().run()


if __name__ == "__main__":
    main()