"""
Phase 7: XGBoost Models
Advanced gradient boosting for improved performance
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, r2_score, mean_absolute_error)
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR


LEAK_COLS = {
    'stock_return_3day', 'abnormal_return', 'label_binary', 'label_median', 'label_tertile',
    'price_before_earnings', 'price_after_earnings',
    'market_date_before', 'market_date_after',
    'is_temporally_valid', 'alignment_timestamp',
    'ticker', 'company_name', 'quarter',
    'transcript_date', 'financial_date', 'fiscal_year', 'fiscal_quarter',
    'financial_date_diff_days', 'financial_DateDiff',
}


class XGBoostModels:

    def __init__(self):
        self.data_dir    = FINAL_DATA_DIR
        self.models_dir  = MODELS_DIR
        self.results_dir = RESULTS_DIR / 'xgboost_models'
        self.log_file    = LOGS_DIR / 'models' / 'xgboost_models.log'
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.all_results = {}

    def load_data(self):
        print("=" * 60)
        print("PHASE 7: XGBOOST MODELS")
        print("=" * 60)
        print()
        print("📂 LOADING DATA...")
        train_df = pd.read_csv(self.data_dir / 'train_data.csv')
        test_df  = pd.read_csv(self.data_dir / 'test_data.csv')
        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        print(f"✅ Train: {len(train_df)} samples")
        print(f"✅ Test:  {len(test_df)} samples")
        print(f"✅ Features: {len(feature_info['feature_columns'])}")
        print()
        return train_df, test_df, feature_info

    def get_feature_sets(self, feature_info, train_df):
        all_cols = (set(feature_info['feature_columns']) & set(train_df.columns)) - LEAK_COLS
        removed  = (set(feature_info['feature_columns']) & set(train_df.columns)) & LEAK_COLS
        if removed:
            print(f"⚠️  Removed {len(removed)} leak/invalid columns from features:")
            for c in sorted(removed):
                print(f"      - {c}")
            print()

        financial = sorted([c for c in all_cols if c.startswith('financial_') or c.startswith('fin_')])
        sentiment = sorted([c for c in all_cols if any(c.startswith(p) for p in [
            'sentiment_', 'mgmt_sentiment_', 'analyst_sentiment_',
            'lm_', 'mgmt_lm_', 'analyst_lm_',
            'word_count', 'question_count', 'lexical_diversity',
            'mgmt_word_count', 'analyst_word_count',
            'lm_sentiment_divergence', 'uncertainty_divergence',
            'mgmt_avg_word_length', 'avg_word_length', 'unique_words',
        ])])
        combined = sorted([c for c in all_cols if not c.startswith('embedding_')])
        full     = sorted(list(all_cols))

        print(f"📊 Feature sets (after leak removal):")
        print(f"   financial_only   : {len(financial)} features")
        print(f"   sentiment_only   : {len(sentiment)} features")
        print(f"   combined         : {len(combined)} features")
        print(f"   full_with_finbert: {len(full)} features")
        print()
        return {'financial_only': financial, 'sentiment_only': sentiment,
                'combined': combined, 'full_with_finbert': full}

    def clean_and_scale(self, X_train, X_test):
        X_train = np.where(np.isinf(X_train), np.nan, X_train)
        X_test  = np.where(np.isinf(X_test),  np.nan, X_test)
 
        medians = np.nanmedian(X_train, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
 
        def _fill(X):
            nan_mask = np.isnan(X)
            if nan_mask.any():
                for j in np.where(nan_mask.any(axis=0))[0]:
                    X[nan_mask[:, j], j] = medians[j]
            return X
 
        X_train = _fill(X_train.copy())
        X_test  = _fill(X_test.copy())
 
        col_std = X_train.std(axis=0)
        keep    = col_std > 0
        if not keep.all():
            print(f"   ⚠️  Dropping {(~keep).sum()} constant/zero-variance column(s)")
            X_train = X_train[:, keep]
            X_test  = X_test[:, keep]
 
        X_train = np.clip(X_train, -1e6, 1e6)
        X_test  = np.clip(X_test,  -1e6, 1e6)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)
 
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)
 
        return X_train, X_test
 

    def prepare(self, train_df, test_df, cols):
        """
        Prepare features with robust data type handling
        """
        # Keep only columns that exist in both datasets
        cols = [c for c in cols if c in train_df.columns and c in test_df.columns]
        
        # Keep only numeric columns
        numeric_cols = []
        for c in cols:
            if pd.api.types.is_numeric_dtype(train_df[c]):
                numeric_cols.append(c)
        
        dropped = sorted(set(cols) - set(numeric_cols))
        if dropped:
            print(f"   ⚠️  Dropping {len(dropped)} non-numeric columns:")
            for c in dropped[:5]:  # Show first 5
                print(f"      - {c} (type: {train_df[c].dtype})")
            if len(dropped) > 5:
                print(f"      ... and {len(dropped)-5} more")
            print()
        
        if len(numeric_cols) == 0:
            print("   ❌ No numeric columns remaining!")
            return np.array([]), np.array([])
        
        X_train = train_df[numeric_cols].values.astype(np.float64)
        X_test  = test_df[numeric_cols].values.astype(np.float64)
        
        return self.clean_and_scale(X_train, X_test)

    def train_xgboost_classifier(self, X_train, y_train, X_test, y_test):
        neg = int(np.sum(y_train == 0))
        pos = int(np.sum(y_train == 1))
        spw = neg / pos if pos > 0 else 1.0
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc',
            'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 200,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_weight': 5, 'gamma': 0.2,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'scale_pos_weight': spw, 'random_state': 42, 'n_jobs': -1,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        return model, {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }

    def train_xgboost_regressor(self, X_train, y_train, X_test, y_test):
        # Sanity check: if target variance is near zero the model cannot learn
        y_std = float(np.std(y_train))
        naive_rmse = float(np.sqrt(mean_squared_error(y_test, np.full_like(y_test, y_train.mean()))))
        if y_std < 1e-6:
            print("   ⚠️  Target variance ≈ 0 — regressor will predict a constant.")

        params = {
            'objective': 'reg:squarederror', 'eval_metric': 'rmse',
            'max_depth': 4, 'learning_rate': 0.05, 'n_estimators': 200,
            'subsample': 0.8, 'colsample_bytree': 0.8,
            'min_child_weight': 5, 'gamma': 0.2,
            'reg_alpha': 0.5, 'reg_lambda': 2.0,
            'random_state': 42, 'n_jobs': -1,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        return model, {
            'rmse':          rmse,
            'mae':           float(mean_absolute_error(y_test, y_pred)),
            'r2':            float(r2_score(y_test, y_pred)),
            'direction':     float(np.mean((y_test > 0) == (y_pred > 0))),
            'naive_rmse':    naive_rmse,
            'vs_naive_rmse': float(rmse - naive_rmse),   # negative = better than naive
            'target_std_train': y_std,
        }

    def run_experiment(self, label, feature_key, train_df, test_df, feature_sets):
        print("=" * 60)
        print(label)
        print("=" * 60)
        cols = feature_sets[feature_key]
        print(f"   Features: {len(cols)}")
        print()
        if len(cols) == 0:
            print("   ⚠️  No features found — skipping.")
            print()
            return None

        X_train, X_test = self.prepare(train_df, test_df, cols)
        y_train_clf = train_df['label_binary'].values
        y_test_clf  = test_df['label_binary'].values
        y_train_reg = train_df['abnormal_return'].values
        y_test_reg  = test_df['abnormal_return'].values

        print("   Training XGBoost Classifier...")
        _, clf_results = self.train_xgboost_classifier(X_train, y_train_clf, X_test, y_test_clf)
        print(f"      ROC-AUC:  {clf_results['roc_auc']:.1%}  ← PRIMARY METRIC")
        print(f"      Accuracy: {clf_results['accuracy']:.1%}")
        print(f"      F1:       {clf_results['f1']:.1%}")
        print()

        print("   Training XGBoost Regressor...")
        _, reg_results = self.train_xgboost_regressor(X_train, y_train_reg, X_test, y_test_reg)
        print(f"      RMSE:       {reg_results['rmse']:.4f}")
        print(f"      Naive RMSE: {reg_results['naive_rmse']:.4f}  (predict-mean baseline)")
        print(f"      vs Naive:   {reg_results['vs_naive_rmse']:+.4f}  ({'better' if reg_results['vs_naive_rmse'] < 0 else 'worse'} than naive)")
        print(f"      R²:         {reg_results['r2']:.4f}")
        print(f"      Direction:  {reg_results['direction']:.1%}")
        print(f"      Target σ (train): {reg_results['target_std_train']:.4f}")
        print()

        return {'feature_set': feature_key, 'n_features': len(cols),
                'classifier': clf_results, 'regressor': reg_results}

    def display_summary(self):
        print("=" * 60)
        print("📊 XGBOOST RESULTS SUMMARY")
        print("=" * 60)
        print()
        print("CLASSIFICATION (ROC-AUC):")
        print(f"{'Feature Set':<25} {'ROC-AUC':<12} {'Accuracy':<12} {'F1'}")
        print("-" * 65)
        for key, result in self.all_results.items():
            clf = result['classifier']
            print(f"{key.replace('_',' ').title():<25} {clf['roc_auc']:<12.1%} {clf['accuracy']:<12.1%} {clf['f1']:.1%}")
        print()
        print("REGRESSION:")
        print(f"{'Feature Set':<25} {'RMSE':<12} {'R²':<12} {'Direction'}")
        print("-" * 60)
        for key, result in self.all_results.items():
            reg = result['regressor']
            print(f"{key.replace('_',' ').title():<25} {reg['rmse']:<12.4f} {reg['r2']:<12.4f} {reg['direction']:.1%}")
        print()
        best_key = max(self.all_results.items(), key=lambda x: x[1]['classifier']['roc_auc'])[0]
        print(f"🏆 BEST MODEL: {best_key.replace('_', ' ').title()}")
        print(f"   ROC-AUC: {self.all_results[best_key]['classifier']['roc_auc']:.1%}")
        print()

    def save_results(self):
        print("=" * 60)
        print("💾 SAVING RESULTS")
        print("=" * 60)
        print()
        json_results = {}
        for key, result in self.all_results.items():
            json_results[key] = {
                'feature_set': result['feature_set'],
                'n_features':  result['n_features'],
                'classifier':  {k: v for k, v in result['classifier'].items() if k != 'confusion_matrix'},
                'regressor':   result['regressor'],
            }
        output_file = self.results_dir / 'phase7_xgboost_results.json'
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"✅ Results saved: {output_file.name}")
        print()
        print("=" * 60)
        print("✅ PHASE 7 COMPLETE")
        print("=" * 60)
        print()

    def run(self):
        train_df, test_df, feature_info = self.load_data()
        feature_sets = self.get_feature_sets(feature_info, train_df)
        experiments = [
            ('7A: XGBOOST — FINANCIAL FEATURES ONLY',     'financial_only'),
            ('7B: XGBOOST — SENTIMENT FEATURES ONLY',     'sentiment_only'),
            ('7C: XGBOOST — COMBINED FEATURES',           'combined'),
            ('7D: XGBOOST — FULL FEATURES (with FinBERT)','full_with_finbert'),
        ]
        for label, key in experiments:
            result = self.run_experiment(label, key, train_df, test_df, feature_sets)
            if result is not None:
                self.all_results[key] = result
        self.display_summary()
        self.save_results()
        return self.all_results


def main():
    XGBoostModels().run()

if __name__ == "__main__":
    main()