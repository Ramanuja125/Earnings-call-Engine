"""
Phase 9: Agentic Feature Engineering
Autonomous AI agent that discovers and evaluates new features
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import xgboost as xgb
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, RESULTS_DIR, LOGS_DIR


# ---------------------------------------------------------------------------
# Leak patterns applied consistently everywhere
# ---------------------------------------------------------------------------
LEAK_PATTERNS = [
    'DateDiff',
    'financial_DateDiff',
    'price_after',
    'price_before',
    'stock_return',
    'alignment_timestamp',
    'label_binary',
    'label_median',
    'label_tertile',
    'abnormal_return',
    'is_temporally_valid',
    'ticker',
    'company_name',
    'quarter',
    'transcript_date',
    'financial_date',
]


def _is_clean_numeric(series: pd.Series) -> bool:
    """Return True only when a column is genuinely numeric (no object dtype)."""
    return pd.api.types.is_numeric_dtype(series)


def _clean_array(X: np.ndarray) -> np.ndarray:
    """Replace inf / -inf / nan with 0."""
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def _filter_base_features(df: pd.DataFrame, candidates: list) -> list:
    """
    Return only columns that:
      - exist in df
      - are genuinely numeric
      - do not match any leak pattern
    """
    return [
        f for f in candidates
        if f in df.columns
        and _is_clean_numeric(df[f])
        and not any(pat in f for pat in LEAK_PATTERNS)
    ]


class AgenticFeatureEngineer:
    """
    Autonomous AI agent for discovering and engineering features.
    """

    def __init__(self):
        self.data_dir = FINAL_DATA_DIR
        self.results_dir = RESULTS_DIR / 'agentic_features'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = LOGS_DIR / 'agentic' / 'feature_engineering.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        self.discovered_features = []
        self.feature_scores = {}
        self.best_features = []

        self.agent_history = {
            'iterations': [],
            'best_score_per_iteration': [],
            'features_tested': 0,
            'features_kept': 0,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self):
        """Load preprocessed data."""
        print("=" * 60)
        print("PHASE 9: AGENTIC FEATURE ENGINEERING")
        print("=" * 60)
        print()
        print("🤖 AI AGENT INITIALIZING...")
        print()
        print("📂 Loading data...")

        df = pd.read_csv(self.data_dir / 'final_dataset.csv')
        # Convert every column to numeric where possible; leave the rest as-is.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)

        print(f"✅ Loaded dataset: {len(df)} records")
        print(f"✅ Existing features: {len(feature_info['feature_columns'])}")
        print()

        return df, feature_info

    # ------------------------------------------------------------------
    # Feature grouping
    # ------------------------------------------------------------------

    def get_base_features(self, df: pd.DataFrame, feature_info: dict) -> dict:
        """Categorise base features into named groups."""
        all_cols = set(feature_info['feature_columns']) & set(df.columns)

        financial  = [c for c in all_cols if c.startswith('financial_')]
        sentiment  = [c for c in all_cols if 'sentiment' in c.lower() or c.startswith('lm_')]
        text_stats = [c for c in all_cols if any(x in c for x in ['word_count', 'lexical', 'question'])]

        # Exclude embedding columns from the generic "all_base" bucket
        embedding_cols = {c for c in all_cols if c.startswith('embedding_')}
        all_base = list(all_cols - embedding_cols)

        return {
            'financial':  financial,
            'sentiment':  sentiment,
            'text_stats': text_stats,
            'market':     [],          # placeholder; extend if market data added
            'all_base':   all_base,
        }

    # ------------------------------------------------------------------
    # Strategy 1 – interaction features
    # ------------------------------------------------------------------

    def generate_interaction_features(self, df: pd.DataFrame, feature_groups: dict) -> dict:
        """AGENT STRATEGY 1: Financial × Sentiment interactions."""
        print("=" * 60)
        print("🤖 AGENT STRATEGY 1: INTERACTION DISCOVERY")
        print("=" * 60)
        print()
        print("🔍 Agent is exploring feature interactions...")
        print()

        new_features = {}

        # Financial × Sentiment
        print("Testing: Financial × Sentiment interactions")
        fin_cols  = _filter_base_features(df, feature_groups['financial'][:10])
        sent_cols = _filter_base_features(df, feature_groups['sentiment'][:5])

        for fin_feat in tqdm(fin_cols, desc="Financial features"):
            for sent_feat in sent_cols:
                try:
                    feat_name = f"interact_{fin_feat}x{sent_feat}"
                    new_features[feat_name] = df[fin_feat] * df[sent_feat]
                except Exception:
                    continue

        print(f"  Created {len(new_features)} interaction features")
        print()

        # Market × Text (no-op until market features are present)
        print("Testing: Market × Text statistics interactions")
        market_cols = _filter_base_features(df, feature_groups['market'])
        text_cols   = _filter_base_features(df, feature_groups['text_stats'])
        count_before = len(new_features)

        for mkt_feat in market_cols:
            for txt_feat in text_cols:
                try:
                    feat_name = f"interact_{mkt_feat}x{txt_feat}"
                    new_features[feat_name] = df[mkt_feat] * df[txt_feat]
                except Exception:
                    continue

        print(f"  Total interaction features: {len(new_features)}")
        print()

        return new_features

    # ------------------------------------------------------------------
    # Strategy 2 – polynomial features
    # ------------------------------------------------------------------

    def generate_polynomial_features(self, df: pd.DataFrame, feature_groups: dict) -> dict:
        """AGENT STRATEGY 2: Polynomial transformations (x², √x, log x)."""
        print("=" * 60)
        print("🤖 AGENT STRATEGY 2: POLYNOMIAL DISCOVERY")
        print("=" * 60)
        print()
        print("🔍 Agent is exploring polynomial transformations...")
        print()

        new_features = {}
        top_financial = _filter_base_features(df, feature_groups['financial'][:15])

        print("Testing: Polynomial features (x², √x, log x)")
        for feat in tqdm(top_financial, desc="Polynomial"):
            try:
                col = df[feat]

                # x²
                new_features[f"poly_{feat}_sq"] = col ** 2

                # √x  (only for non-negative columns)
                if (col >= 0).all():
                    new_features[f"poly_{feat}_sqrt"] = np.sqrt(col)

                # log(x + 1)  (only for strictly positive columns)
                if (col > 0).all():
                    new_features[f"poly_log_{feat}"] = np.log1p(col)

            except Exception:
                continue

        print(f"  Created {len(new_features)} polynomial features")
        print()

        return new_features

    # ------------------------------------------------------------------
    # Strategy 3 – ratio features
    # ------------------------------------------------------------------

    def generate_ratio_features(self, df: pd.DataFrame, feature_groups: dict) -> dict:
        """AGENT STRATEGY 3: Ratio features."""
        print("=" * 60)
        print("🤖 AGENT STRATEGY 3: RATIO DISCOVERY")
        print("=" * 60)
        print()
        print("🔍 Agent is exploring ratio relationships...")
        print()

        new_features = {}

        # Financial ratios
        financial = _filter_base_features(df, feature_groups['financial'][:10])
        print("Testing: Financial ratio features")
        for i, feat1 in enumerate(financial):
            for feat2 in financial[i + 1: i + 4]:
                try:
                    denominator = df[feat2].replace(0, np.nan)
                    feat_name = f"ratio_{feat1}_div_{feat2}"
                    new_features[feat_name] = df[feat1] / denominator
                except Exception:
                    continue

        # Sentiment ratios
        sentiment = _filter_base_features(df, feature_groups['sentiment'][:8])
        print("Testing: Sentiment ratio features")
        for i, feat1 in enumerate(sentiment):
            for feat2 in sentiment[i + 1: i + 3]:
                try:
                    denominator = df[feat2].replace(0, np.nan)
                    feat_name = f"ratio_{feat1}_div_{feat2}"
                    new_features[feat_name] = df[feat1] / denominator
                except Exception:
                    continue

        print(f"  Created {len(new_features)} ratio features")
        print()

        return new_features

    # ------------------------------------------------------------------
    # Strategy 4 – aggregation features
    # ------------------------------------------------------------------

    def generate_aggregation_features(self, df: pd.DataFrame, feature_groups: dict) -> dict:
        """AGENT STRATEGY 4: Row-wise aggregations across feature groups."""
        print("=" * 60)
        print("🤖 AGENT STRATEGY 4: AGGREGATION DISCOVERY")
        print("=" * 60)
        print()
        print("🔍 Agent is exploring feature aggregations...")
        print()

        new_features = {}

        # Financial aggregations
        financial = _filter_base_features(df, feature_groups['financial'])
        if financial:
            df_fin = df[financial].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
            new_features['agg_financial_mean'] = df_fin.mean(axis=1)
            new_features['agg_financial_std']  = df_fin.std(axis=1)
            new_features['agg_financial_max']  = df_fin.max(axis=1)
            new_features['agg_financial_min']  = df_fin.min(axis=1)

        # Sentiment aggregations (score columns only)
        sentiment_scores = _filter_base_features(
            df, [c for c in feature_groups['sentiment'] if 'score' in c]
        )
        if sentiment_scores:
            df_sent = df[sentiment_scores].copy().apply(pd.to_numeric, errors='coerce').fillna(0)
            new_features['agg_sentiment_mean']  = df_sent.mean(axis=1)
            new_features['agg_sentiment_std']   = df_sent.std(axis=1)
            new_features['agg_sentiment_range'] = df_sent.max(axis=1) - df_sent.min(axis=1)

        print(f"  Created {len(new_features)} aggregation features")
        print()

        return new_features

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_features(
        self,
        df: pd.DataFrame,
        new_features: dict,
        target: np.ndarray,
    ):
        """Score every discovered feature with mutual information."""
        print("=" * 60)
        print("🤖 AGENT EVALUATION: SCORING FEATURES")
        print("=" * 60)
        print()
        print("📊 Agent is evaluating discovered features...")
        print()

        new_features_df = pd.DataFrame(new_features, index=df.index)

        # ── Sanitise: inf → nan → fill ────────────────────────────────
        # Step 1: replace inf/-inf with nan
        new_features_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Step 2: fill nan with column median; if median is also nan
        #         (entire column is nan — happens with tiny datasets),
        #         fall back to 0.0 so the column is still usable.
        for col in new_features_df.columns:
            if new_features_df[col].isna().any():
                med = new_features_df[col].median()
                fill_val = med if not np.isnan(med) else 0.0
                new_features_df[col].fillna(fill_val, inplace=True)

        # Step 3: drop columns that are still all-NaN or constant
        #         (constant columns cause mutual_info to return 0 anyway,
        #          and can confuse sklearn with tiny datasets)
        before = len(new_features_df.columns)
        new_features_df.dropna(axis=1, how='all', inplace=True)
        constant_cols = [c for c in new_features_df.columns
                         if new_features_df[c].nunique() <= 1]
        if constant_cols:
            new_features_df.drop(columns=constant_cols, inplace=True)
        dropped = before - len(new_features_df.columns)
        if dropped:
            print(f"   ℹ️  Dropped {dropped} all-NaN or constant columns")

        if new_features_df.empty or len(new_features_df.columns) == 0:
            print("⚠️  No usable features found after sanitisation. Exiting phase 9.")
            return pd.DataFrame(), {}

        # Step 4: final safety net — replace any surviving nan/inf with 0
        new_features_df = new_features_df.apply(
            pd.to_numeric, errors='coerce').fillna(0.0)

        # Step 5: n_neighbors must be < n_samples; scale down for tiny datasets
        n_samples = len(new_features_df)
        n_neighbors = min(5, max(1, n_samples // 5))

        print(f"Computing mutual information scores ({n_samples} records, "
              f"{len(new_features_df.columns)} candidate features)...")

        try:
            mi_scores = mutual_info_classif(
                new_features_df,
                target,
                discrete_features=False,
                random_state=42,
                n_neighbors=n_neighbors,
            )
        except Exception as e:
            print(f"❌ Error in mutual information scoring: {e}")
            return pd.DataFrame(), {}

        feature_scores = dict(zip(new_features_df.columns, mi_scores))
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        print()
        print("🏆 TOP 20 DISCOVERED FEATURES:")
        print()

        for i, (feat, score) in enumerate(sorted_features[:20], 1):
            print(f"  {i:2d}. {feat:<60} Score: {score:.4f}")
            self.discovered_features.append({'rank': i, 'feature': feat, 'score': score})

        print()

        # Select top features: score threshold + leak filter
        threshold_score = 0.02
        top_n = 15

        self.best_features = [
            feat for feat, score in sorted_features
            if score > threshold_score
            and not any(pat in feat for pat in LEAK_PATTERNS)
        ][:top_n]

        print(f"✅ Agent selected {len(self.best_features)} high-value features")
        print(f"   (Leak patterns filtered: {len(LEAK_PATTERNS)} patterns blocked)")
        print(f"   (Threshold: score > {threshold_score}, max {top_n})")
        print()

        self.feature_scores = feature_scores
        self.agent_history['features_kept'] = len(self.best_features)

        if not self.best_features:
            print("⚠️  No features passed the threshold.")
            print(f"   Dataset has {len(df)} records — consider lowering threshold_score.")
            print(f"   Returning all scored features instead (top {top_n}).")
            # Fallback: return top_n regardless of threshold
            self.best_features = [f for f, _ in sorted_features[:top_n]
                                  if not any(pat in f for pat in LEAK_PATTERNS)]
            if not self.best_features:
                return pd.DataFrame(), feature_scores

        return new_features_df[self.best_features].copy(), feature_scores

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_features_with_model(
        self,
        df: pd.DataFrame,
        base_feature_candidates: list,
        new_features_df: pd.DataFrame,
        target: np.ndarray,
    ):
        """Compare baseline vs. enhanced XGBoost with time-series CV."""
        print("=" * 60)
        print("🤖 AGENT VALIDATION: MODEL TESTING")
        print("=" * 60)
        print()
        print("🔬 Agent is validating features with XGBoost...")
        print()

        # ----------------------------------------------------------------
        # Build the clean base-feature matrix in ONE step:
        #   ✓ column exists in df
        #   ✓ genuinely numeric dtype
        #   ✓ not a leak column
        # ----------------------------------------------------------------
        base_features = _filter_base_features(df, base_feature_candidates)

        dropped = len(base_feature_candidates) - len(base_features)
        if dropped:
            print(f"⚠️  Dropped {dropped} columns (non-numeric or leak patterns)")

        if not base_features:
            raise ValueError("No valid base features remain after filtering.")

        X_base = _clean_array(df[base_features].values)
        scaler_base = StandardScaler()
        X_base_scaled = scaler_base.fit_transform(X_base)

        tscv = TimeSeriesSplit(n_splits=3)

        # Baseline
        print("Testing baseline (human-designed features only)...")
        model_base = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=100,
            random_state=42, n_jobs=1,
            eval_metric='logloss',
        )
        scores_base = cross_val_score(
            model_base, X_base_scaled, target,
            cv=tscv, scoring='roc_auc', n_jobs=1,
        )
        baseline_auc = scores_base.mean()
        print(f"  Baseline AUC: {baseline_auc:.3f} (±{scores_base.std():.3f})")
        print()

        # Enhanced
        print("Testing enhanced (baseline + agent-discovered features)...")

        # Sanitise the new features one more time before stacking
        new_vals = _clean_array(new_features_df.values)
        X_enhanced = np.hstack([X_base_scaled, new_vals])

        model_enhanced = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=100,
            random_state=42, n_jobs=1,
            eval_metric='logloss',
        )
        scores_enhanced = cross_val_score(
            model_enhanced, X_enhanced, target,
            cv=tscv, scoring='roc_auc', n_jobs=1,
        )
        enhanced_auc = scores_enhanced.mean()
        print(f"  Enhanced AUC: {enhanced_auc:.3f} (±{scores_enhanced.std():.3f})")
        print()

        improvement = enhanced_auc - baseline_auc
        improvement_pct = (improvement / baseline_auc) * 100 if baseline_auc else 0.0

        print("=" * 60)
        print("📊 AGENTIC FEATURE ENGINEERING RESULTS")
        print("=" * 60)
        print()
        print(f"Baseline (human-designed):  {baseline_auc:.3f}")
        print(f"Enhanced (+ AI-discovered): {enhanced_auc:.3f}")
        print(f"Improvement:                {improvement:+.3f} ({improvement_pct:+.1f}%)")
        print()

        if improvement > 0:
            print("✅ Agent successfully discovered valuable features!")
        else:
            print("⚠️  Agent-discovered features did not improve performance")
        print()

        self.agent_history['baseline_auc']    = float(baseline_auc)
        self.agent_history['enhanced_auc']    = float(enhanced_auc)
        self.agent_history['improvement']     = float(improvement)
        self.agent_history['improvement_pct'] = float(improvement_pct)

        return baseline_auc, enhanced_auc, improvement

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_discovered_features(self, new_features_df: pd.DataFrame):
        """Save agent-discovered features and metadata."""
        print("=" * 60)
        print("💾 SAVING AGENT DISCOVERIES")
        print("=" * 60)
        print()

        features_path = self.results_dir / 'agent_discovered_features.csv'
        new_features_df.to_csv(features_path, index=False)
        print(f"✅ Discovered features saved: {features_path.name}")
        print(f"   Count: {len(new_features_df.columns)} features")

        if self.discovered_features:
            rankings_df = pd.DataFrame(self.discovered_features)
            rankings_path = self.results_dir / 'feature_rankings.csv'
            rankings_df.to_csv(rankings_path, index=False)
            print(f"✅ Feature rankings saved: {rankings_path.name}")

        history_path = self.results_dir / 'agent_learning_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.agent_history, f, indent=2)
        print(f"✅ Agent history saved: {history_path.name}")
        print()

    def generate_report(self):
        """Generate and save a JSON summary report."""
        print("=" * 60)
        print("📝 GENERATING AGENT REPORT")
        print("=" * 60)
        print()

        report = {
            'phase':               'Phase 9: Agentic Feature Engineering',
            'timestamp':           datetime.now().isoformat(),
            'agent_strategies':    [
                'Interaction Discovery',
                'Polynomial Discovery',
                'Ratio Discovery',
                'Aggregation Discovery',
            ],
            'features_tested':     self.agent_history.get('features_tested', 0),
            'features_discovered': len(self.best_features),
            'baseline_auc':        self.agent_history.get('baseline_auc', 0),
            'enhanced_auc':        self.agent_history.get('enhanced_auc', 0),
            'improvement':         self.agent_history.get('improvement', 0),
            'improvement_pct':     self.agent_history.get('improvement_pct', 0),
            'top_10_features':     self.discovered_features[:10],
        }

        report_path = self.results_dir / 'phase9_agentic_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✅ Agent report saved: {report_path.name}")
        print()
        print("🤖 AGENTIC AI SUMMARY:")
        print()
        print(f"Strategies employed:      {len(report['agent_strategies'])}")
        print(f"Features discovered:      {report['features_discovered']}")
        print(f"Performance improvement:  {report['improvement_pct']:.1f}%")
        print()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self):
        """Execute the complete agentic feature-engineering workflow."""

        df, feature_info = self.load_data()
        feature_groups   = self.get_base_features(df, feature_info)

        print("📊 Base Feature Groups:")
        for name, cols in feature_groups.items():
            if name != 'all_base':
                print(f"   {name.capitalize()}: {len(cols)}")
        print()

        # Discover features via four independent strategies
        all_new_features: dict = {}
        all_new_features.update(self.generate_interaction_features(df, feature_groups))
        all_new_features.update(self.generate_polynomial_features(df, feature_groups))
        all_new_features.update(self.generate_ratio_features(df, feature_groups))
        all_new_features.update(self.generate_aggregation_features(df, feature_groups))

        self.agent_history['features_tested'] = len(all_new_features)

        print("🤖 AGENT DISCOVERY SUMMARY:")
        print(f"   Total features created: {len(all_new_features)}")
        print()

        target = df['label_binary'].values

        # Evaluate
        new_features_df, feature_scores = self.evaluate_features(df, all_new_features, target)

        if new_features_df.empty:
            print("⚠️  No usable features found. Exiting phase 9.")
            return False

        # Validate with a real model
        self.validate_features_with_model(
            df, feature_groups['all_base'], new_features_df, target
        )

        # Persist
        self.save_discovered_features(new_features_df)
        self.generate_report()

        print("=" * 60)
        print("✅ PHASE 9 COMPLETE")
        print("=" * 60)
        print()
        print(f"📁 Results saved to: {self.results_dir}/")
        print()
        print("✅ Ready for Phase 10: Agent-Enhanced Modeling")
        print()

        return True


def main():
    agent = AgenticFeatureEngineer()
    agent.run()


if __name__ == "__main__":
    main()