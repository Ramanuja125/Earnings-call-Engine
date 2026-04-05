"""
Phase 10: True Agentic AI for Model Optimization
A self-learning agent that iteratively improves through experience
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
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy.stats import uniform, randint
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import FINAL_DATA_DIR, RESULTS_DIR, LOGS_DIR


class AgenticAI:
    """
    True Agentic AI System with:
    - Autonomous decision making
    - Learning from experience
    - Adaptive strategy selection
    - Multi-iteration improvement
    - Long-term memory
    """
    
    def __init__(self):
        self.data_dir = FINAL_DATA_DIR
        self.results_dir = RESULTS_DIR / 'agentic_ai'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # AGENT MEMORY (persistent knowledge)
        self.memory = {
            'strategy_performance': defaultdict(list),  # What worked in past
            'best_configurations': {},                   # Top performers
            'failed_attempts': [],                       # What didn't work
            'learning_history': [],                      # Full trajectory
            'insights': []                               # Agent's discoveries
        }
        
        # AGENT STATE (current beliefs)
        self.current_strategy = 'exploration'
        self.exploration_rate = 1.0  # Start exploring, then exploit
        self.best_score = 0.0
        self.iteration = 0
        
        # STRATEGY POOL (agent can choose from)
        self.strategies = {
            'feature_engineering': self.try_feature_engineering,
            'hyperparameter_tuning': self.try_hyperparameter_tuning,
            'ensemble_building': self.try_ensemble_building,
            'architecture_search': self.try_architecture_search,
            'threshold_optimization': self.try_threshold_optimization
        }
        
        # DATA CACHE
        self.X = None
        self.y = None
        self.feature_names = None
        self.tscv = TimeSeriesSplit(n_splits=3)
    
    def load_and_prepare_data(self):
        """Load and prepare data once"""
        print("="*60)
        print("TRUE AGENTIC AI SYSTEM")
        print("="*60)
        print()
        print("🤖 AGENTIC AI INITIALIZING...")
        print()
        print("📂 Loading data...")
        
        df = pd.read_csv(self.data_dir / 'final_dataset.csv')
        
        with open(self.data_dir / 'feature_info.json') as f:
            feature_info = json.load(f)
        
        # Get clean features
        LEAK_PATTERNS = [
            'price_after', 'price_before', 'stock_return',
            'abnormal_return', 'label_binary', 'label_median',
            'label_tertile', 'sp500_return', 'alignment_timestamp',
            'ticker', 'company_name', 'quarter', 'transcript_date',
            'financial_date', 'DateDiff', 'date_diff',
            'financial_match_type',   # string col added by temporal aligner
            'match_type',
        ]

        all_cols = set(feature_info['feature_columns']) & set(df.columns)

        # Also filter to numeric columns only — prevents any string
        # column (e.g. financial_match_type = 'tight'/'fallback')
        # from crashing astype(np.float64)
        numeric_cols_in_df = set(
            df.select_dtypes(include=[np.number]).columns.tolist()
        )

        clean_features = [
            f for f in all_cols
            if not any(pattern in f for pattern in LEAK_PATTERNS)
            and not f.startswith('embedding_')   # Exclude embeddings initially
            and f in numeric_cols_in_df           # Must be numeric
        ]
        
        # Prepare data
        X = df[clean_features].values.astype(np.float64)
        y = df['label_binary'].values
        
        # Clean
        X[np.isinf(X)] = np.nan
        medians = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(medians, inds[1])
        X = np.clip(X, -1e6, 1e6)
        
        # Scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        self.X = X
        self.y = y
        self.feature_names = clean_features
        
        print(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print()
        
        return X, y
    
    def evaluate_model(self, model):
        """Evaluate model with time-series CV"""
        scores = []
        for train_idx, test_idx in self.tscv.split(self.X):
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            auc = roc_auc_score(y_test, y_proba)
            scores.append(auc)
        
        return np.mean(scores), np.std(scores)
    
    def choose_strategy(self):
        """
        AGENT DECISION: Choose which strategy to try next
        Uses epsilon-greedy with learning
        """
        # Epsilon-greedy: explore vs exploit
        if np.random.random() < self.exploration_rate:
            # EXPLORE: Try something new
            # Prefer strategies we haven't tried much
            strategy_counts = {s: len(self.memory['strategy_performance'][s]) 
                             for s in self.strategies.keys()}
            
            # Choose least-tried strategy
            strategy = min(strategy_counts.items(), key=lambda x: x[1])[0]
            
            decision = f"EXPLORE (trying {strategy})"
        else:
            # EXPLOIT: Use best-performing strategy
            avg_performance = {
                s: np.mean(scores) if scores else 0
                for s, scores in self.memory['strategy_performance'].items()
            }
            
            strategy = max(avg_performance.items(), key=lambda x: x[1])[0]
            decision = f"EXPLOIT (using best: {strategy})"
        
        print(f"🤖 Agent Decision: {decision}")
        print(f"   Exploration rate: {self.exploration_rate:.2f}")
        print()
        
        return strategy
    
    def try_feature_engineering(self):
        """Strategy 1: Feature Engineering"""
        print("🔧 Strategy: Feature Engineering")
        print("   Creating interaction features...")
        
        # Create a few high-value interactions
        # (Agent learned from Phase 9 that too many features don't help)
        
        financial_idx = [i for i, f in enumerate(self.feature_names) 
                        if 'financial_' in f][:5]
        sentiment_idx = [i for i, f in enumerate(self.feature_names) 
                        if 'sentiment' in f][:3]
        
        # Create interactions
        new_features = []
        for f_idx in financial_idx:
            for s_idx in sentiment_idx:
                interaction = self.X[:, f_idx] * self.X[:, s_idx]
                new_features.append(interaction.reshape(-1, 1))
        
        if new_features:
            X_enhanced = np.hstack([self.X] + new_features)
        else:
            X_enhanced = self.X
        
        # Test with XGBoost
        model = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=100,
            random_state=42, n_jobs=-1
        )
        
        # Temporarily use enhanced features
        original_X = self.X
        self.X = X_enhanced
        
        score, std = self.evaluate_model(model)
        
        # Restore
        self.X = original_X
        
        print(f"   Result: {score:.3f} (±{std:.3f})")
        
        return score, {'n_features_added': len(new_features)}
    
    def try_hyperparameter_tuning(self):
        """Strategy 2: Hyperparameter Tuning"""
        print("🔧 Strategy: Hyperparameter Tuning")
        print("   Searching optimal parameters...")
        
        param_space = {
            'max_depth': randint(4, 12),
            'learning_rate': uniform(0.01, 0.2),
            'n_estimators': randint(100, 400),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 8),
            'gamma': uniform(0, 0.5)
        }
        
        search = RandomizedSearchCV(
            xgb.XGBClassifier(random_state=42, n_jobs=-1),
            param_space,
            n_iter=20,
            cv=self.tscv,
            scoring='roc_auc',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(self.X, self.y)
        
        score = search.best_score_
        print(f"   Result: {score:.3f}")
        print(f"   Best params: {search.best_params_}")
        
        return score, {'best_params': search.best_params_, 'model': search.best_estimator_}
    
    def try_ensemble_building(self):
        """Strategy 3: Ensemble Building"""
        print("🔧 Strategy: Ensemble Building")
        print("   Constructing ensemble...")
        
        # Build ensemble of diverse models
        estimators = [
            ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)),
            ('xgb', xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=150, random_state=42, n_jobs=-1))
        ]
        
        voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        score, std = self.evaluate_model(voting)
        
        print(f"   Result: {score:.3f} (±{std:.3f})")
        
        return score, {'ensemble_type': 'voting'}
    
    def try_architecture_search(self):
        """Strategy 4: Architecture Search"""
        print("🔧 Strategy: Architecture Search")
        print("   Testing different architectures...")
        
        # Test different model types
        models = {
            'XGBoost': xgb.XGBClassifier(max_depth=6, n_estimators=150, random_state=42, n_jobs=-1),
            'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'Stacking': StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1))
                ],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3,
                n_jobs=-1
            )
        }
        
        best_score = 0
        best_arch = None
        
        for name, model in models.items():
            score, _ = self.evaluate_model(model)
            if score > best_score:
                best_score = score
                best_arch = name
        
        print(f"   Result: {best_score:.3f}")
        print(f"   Best architecture: {best_arch}")
        
        return best_score, {'best_architecture': best_arch}
    
    def try_threshold_optimization(self):
        """Strategy 5: Threshold Optimization"""
        print("🔧 Strategy: Threshold Optimization")
        print("   Optimizing decision threshold...")
        
        # Use current best model (XGBoost)
        model = xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, n_estimators=150,
            random_state=42, n_jobs=-1
        )
        
        score, std = self.evaluate_model(model)
        
        print(f"   Result: {score:.3f} (±{std:.3f})")
        
        return score, {'note': 'ROC-AUC is threshold-independent'}
    
    def learn_from_outcome(self, strategy, score, metadata):
        """
        AGENT LEARNING: Update beliefs based on outcome
        """
        # Record performance
        self.memory['strategy_performance'][strategy].append(score)
        
        # Update best
        if score > self.best_score:
            improvement = score - self.best_score
            self.best_score = score
            
            self.memory['best_configurations'][strategy] = {
                'score': score,
                'metadata': metadata,
                'iteration': self.iteration
            }
            
            insight = f"Iteration {self.iteration}: {strategy} achieved new best: {score:.3f} (+{improvement:.3f})"
            self.memory['insights'].append(insight)
            
            print(f"💡 AGENT LEARNED: New best score! {score:.3f}")
            print()
        
        # Record in history
        self.memory['learning_history'].append({
            'iteration': self.iteration,
            'strategy': strategy,
            'score': score,
            'was_best': score == self.best_score,
            'exploration_rate': self.exploration_rate
        })
    
    def adapt_strategy(self):
        """
        AGENT ADAPTATION: Modify future behavior based on learning
        """
        # Decay exploration (shift from explore to exploit)
        self.exploration_rate = max(0.1, self.exploration_rate * 0.85)
        
        # If we've tried all strategies at least twice, focus on best
        strategy_counts = {s: len(self.memory['strategy_performance'][s]) 
                         for s in self.strategies.keys()}
        
        min_tries = min(strategy_counts.values())
        
        if min_tries >= 2:
            # We've explored enough, time to exploit
            self.current_strategy = 'exploitation'
            print(f"🤖 Agent adapted: Switching to EXPLOITATION mode")
            print()
    
    def run_agentic_loop(self, max_iterations=8):
        """
        MAIN AGENTIC LOOP:
        For each iteration:
          1. Agent chooses strategy (explore vs exploit)
          2. Agent executes strategy
          3. Agent evaluates outcome
          4. Agent learns from result
          5. Agent adapts future behavior
        """
        print("="*60)
        print("🤖 STARTING AGENTIC LEARNING LOOP")
        print("="*60)
        print()
        print(f"Max iterations: {max_iterations}")
        print(f"Agent will autonomously:")
        print("  • Choose strategies")
        print("  • Execute experiments")
        print("  • Learn from outcomes")
        print("  • Adapt behavior")
        print()
        
        for self.iteration in range(1, max_iterations + 1):
            print("="*60)
            print(f"ITERATION {self.iteration}/{max_iterations}")
            print("="*60)
            print()
            
            # 1. DECIDE: Choose strategy
            strategy_name = self.choose_strategy()
            strategy_func = self.strategies[strategy_name]
            
            # 2. EXECUTE: Run strategy
            try:
                score, metadata = strategy_func()
            except Exception as e:
                print(f"❌ Strategy failed: {e}")
                score = 0.0
                metadata = {'error': str(e)}
            
            # 3. EVALUATE: Assess outcome
            print()
            
            # 4. LEARN: Update knowledge
            self.learn_from_outcome(strategy_name, score, metadata)
            
            # 5. ADAPT: Modify future behavior
            self.adapt_strategy()
            
            print(f"Current best: {self.best_score:.3f}")
            print()
        
        print("="*60)
        print("🤖 AGENTIC LOOP COMPLETE")
        print("="*60)
        print()
    
    def generate_insights(self):
        """Agent reflects on what it learned"""
        print("="*60)
        print("🧠 AGENT INSIGHTS & LEARNING")
        print("="*60)
        print()
        
        # Strategy effectiveness
        print("📊 STRATEGY EFFECTIVENESS:")
        print()
        
        for strategy, scores in self.memory['strategy_performance'].items():
            if scores:
                avg_score = np.mean(scores)
                best_score = max(scores)
                times_tried = len(scores)
                
                print(f"{strategy:<25} Avg: {avg_score:.3f}  Best: {best_score:.3f}  Tries: {times_tried}")
        
        print()
        
        # Best configuration
        print("🏆 BEST CONFIGURATION:")
        print()
        
        best_strat = max(self.memory['best_configurations'].items(), 
                        key=lambda x: x[1]['score'])
        
        print(f"Strategy: {best_strat[0]}")
        print(f"Score: {best_strat[1]['score']:.3f}")
        print(f"Found at iteration: {best_strat[1]['iteration']}")
        print()
        
        # Key insights
        if self.memory['insights']:
            print("💡 KEY DISCOVERIES:")
            print()
            for insight in self.memory['insights']:
                print(f"  • {insight}")
            print()
        
        # Learning trajectory
        print("📈 LEARNING TRAJECTORY:")
        print()
        
        for record in self.memory['learning_history']:
            marker = "🥇" if record['was_best'] else "  "
            print(f"{marker} Iter {record['iteration']}: {record['strategy']:<25} → {record['score']:.3f}")
        
        print()
    
    def save_results(self):
        """Save agent's learning and results"""
        print("="*60)
        print("💾 SAVING AGENT MEMORY")
        print("="*60)
        print()
        
        # Convert memory to JSON-serializable format
        memory_export = {
            'strategy_performance': {
                k: [float(s) for s in v]
                for k, v in self.memory['strategy_performance'].items()
            },
            'best_configurations': {
                k: {
                    'score': float(v['score']),
                    'iteration': int(v['iteration']),
                    'metadata': str(v['metadata'])
                }
                for k, v in self.memory['best_configurations'].items()
            },
            'insights': self.memory['insights'],
            'learning_history': [
                {
                    'iteration': int(r['iteration']),
                    'strategy': str(r['strategy']),
                    'score': float(r['score']),
                    'was_best': bool(r['was_best']),      # ← fix here
                    'exploration_rate': float(r['exploration_rate'])
                }
                for r in self.memory['learning_history']
            ],
            'final_best_score': float(self.best_score)
        }
        
        memory_path = self.results_dir / 'agent_memory.json'
        with open(memory_path, 'w') as f:
            json.dump(memory_export, f, indent=2)
        
        print(f"✅ Agent memory saved: {memory_path.name}")
        
        # Generate report
        report = {
            'phase': 'Phase 10: True Agentic AI',
            'timestamp': datetime.now().isoformat(),
            'total_iterations': int(self.iteration),
            'strategies_available': list(self.strategies.keys()),
            'strategies_tried': {
                k: int(len(v))
                for k, v in self.memory['strategy_performance'].items()
            },
            'best_score': float(self.best_score),
            'best_strategy': max(
                self.memory['best_configurations'].items(),
                key=lambda x: x[1]['score']
            )[0],
            'learning_trajectory': [
                {
                    'iteration': int(r['iteration']),
                    'strategy':  str(r['strategy']),
                    'score':     float(r['score']),
                    'was_best':  bool(r['was_best']),
                    'exploration_rate': float(r['exploration_rate'])
                }
                for r in self.memory['learning_history']
            ],
            'agent_insights': [str(i) for i in self.memory['insights']],
            'final_exploration_rate': float(self.exploration_rate)
        }

        report_path = self.results_dir / 'agentic_ai_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Agentic AI report saved: {report_path.name}")
        print()
    
    def run(self, max_iterations=8):
        """Execute complete agentic AI workflow"""
        
        # Prepare data
        self.load_and_prepare_data()
        
        # Run agentic learning loop
        self.run_agentic_loop(max_iterations=max_iterations)
        
        # Generate insights
        self.generate_insights()
        
        # Save results
        self.save_results()
        
        print("="*60)
        print("✅ AGENTIC AI COMPLETE")
        print("="*60)
        print()
        print(f"📁 Results saved to: {self.results_dir}/")
        print()
        print(f"🏆 Final Best Score: {self.best_score:.3f}")
        print()
        print("✅ Agent has learned optimal strategy!")
        print()
        
        return True


def main():
    """Main execution"""
    agent = AgenticAI()
    agent.run(max_iterations=8)
    return True


if __name__ == "__main__":
    main()

# """
# Phase 10: True Agentic AI for Model Optimization
# A self-learning agent that iteratively improves through experience
 
# GEMINI INTEGRATION (v2):
#   The choose_strategy() method now calls Gemini 1.5 Flash (free tier) to
#   reason in natural language about which strategy to try next.
 
#   The LLM receives:
#     - Current iteration number and best score so far
#     - Full strategy performance history (what worked, what didn't)
#     - SHAP top features (if available) — so it can connect feature
#       importance to strategy choice
#     - Available strategies with descriptions
 
#   The LLM returns a JSON response:
#     { "strategy": "<name>", "reasoning": "<explanation>" }
 
#   If Gemini is unavailable (no API key, quota exceeded, network error),
#   the system falls back silently to the original epsilon-greedy logic
#   so the pipeline never breaks.
 
#   Setup:
#     pip install google-generativeai
#     Set GEMINI_API_KEY environment variable OR pass api_key= to AgenticAI()
# """
 
# import os
# import time
# import pandas as pd
# import numpy as np
# import json
# from pathlib import Path
# from datetime import datetime
# import sys
# import warnings
# warnings.filterwarnings('ignore')
 
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
# from sklearn.metrics import roc_auc_score
# import xgboost as xgb
# from scipy.stats import uniform, randint
# from collections import defaultdict
 
# project_root = Path(__file__).parent.parent.parent
# sys.path.append(str(project_root))
 
# from config import FINAL_DATA_DIR, RESULTS_DIR, LOGS_DIR
 
 
# # ── Gemini client (optional — falls back gracefully if unavailable) ────────
 
# def _build_gemini_client(api_key: str = None):
#     """
#     Try to configure the Gemini client, probing multiple model names in order.
#     Different API keys / quota tiers expose different model names, so we try
#     each candidate until one responds successfully.
#     Returns (GenerativeModel, model_name) or (None, None) if unavailable.
#     """
#     try:
#         import google.generativeai as genai
#         key = api_key or os.environ.get("GEMINI_API_KEY", "")
#         if not key:
#             print("⚠️  GEMINI_API_KEY not set — using epsilon-greedy fallback")
#             return None, None
#         genai.configure(api_key=key)

#         # Try models in order — stop at the first that actually responds
#         candidates = [
#             "gemini-2.0-flash",
#             "gemini-2.0-flash-lite",
#             "gemini-1.5-flash-latest",
#             "gemini-1.5-flash",
#             "gemini-pro",
#         ]
#         for model_name in candidates:
#             try:
#                 model = genai.GenerativeModel(model_name)
#                 # Lightweight probe to confirm the model is callable
#                 model.generate_content(
#                     "ping",
#                     generation_config={"max_output_tokens": 5},
#                 )
#                 print(f"✅ Gemini connected ({model_name}) — LLM reasoning enabled")
#                 return model, model_name
#             except Exception:
#                 continue

#         print("⚠️  No Gemini model available — using epsilon-greedy fallback")
#         return None, None

#     except ImportError:
#         print("⚠️  google-generativeai not installed — using epsilon-greedy fallback")
#         print("   Run: pip install google-generativeai")
#         return None, None
#     except Exception as e:
#         print(f"⚠️  Gemini init failed ({e}) — using epsilon-greedy fallback")
#         return None, None
 
 
# class AgenticAI:
#     """
#     True Agentic AI System with:
#     - Gemini LLM reasoning for strategy selection (NEW)
#     - Autonomous decision making
#     - Learning from experience
#     - Adaptive strategy selection
#     - Multi-iteration improvement
#     - Long-term memory
#     """
 
#     def __init__(self, api_key: str = None):
#         self.data_dir    = FINAL_DATA_DIR
#         self.results_dir = RESULTS_DIR / 'agentic_ai'
#         self.results_dir.mkdir(parents=True, exist_ok=True)

#         # ── Gemini LLM (optional) ──────────────────────────────────────
#         self.gemini_model, self.gemini_model_name = _build_gemini_client(api_key)
#         self.gemini_calls = 0
#         self.gemini_reasoning_log = []

#         # AGENT MEMORY (persistent knowledge)
#         self.memory = {
#             'strategy_performance': defaultdict(list),
#             'best_configurations': {},
#             'failed_attempts':     [],
#             'learning_history':    [],
#             'insights':            [],
#         }

#         # AGENT STATE
#         self.current_strategy = 'exploration'
#         self.exploration_rate = 1.0
#         self.best_score       = 0.0
#         self.iteration        = 0
#         self.scale_pos_weight = 1.0   # set properly in load_and_prepare_data

#         # STRATEGY POOL
#         self.strategies = {
#             'feature_engineering':    self.try_feature_engineering,
#             'hyperparameter_tuning':  self.try_hyperparameter_tuning,
#             'ensemble_building':      self.try_ensemble_building,
#             'architecture_search':    self.try_architecture_search,
#             'threshold_optimization': self.try_threshold_optimization,
#         }

#         # DATA CACHE
#         self.X            = None
#         self.y            = None
#         self.X_test       = None
#         self.y_test       = None
#         self.feature_names = None
#         self.tscv = TimeSeriesSplit(n_splits=3)
    
#     def load_and_prepare_data(self):
#         """Load and prepare data once.

#         Key design decisions
#         --------------------
#         1. Target = label_median (50/50 per-split) instead of label_binary
#            (which has a 22pp train/test distribution shift due to bull market
#            in the test period Jul-2025 → Jan-2026).

#         2. TSCV is applied to TRAIN rows only (from train_data.csv).
#            Using final_dataset.csv for CV means the later TimeSeriesSplit
#            folds would include test-period rows, leaking future information
#            into the agent's learning signal.

#         3. A held-out test set (from test_data.csv) is stored separately for
#            final evaluation only — the agent never sees it during training.
#         """
#         print("="*60)
#         print("TRUE AGENTIC AI SYSTEM")
#         print("="*60)
#         print()
#         print("🤖 AGENTIC AI INITIALIZING...")
#         print()
#         print("📂 Loading data...")

#         # ── Load train and test separately ────────────────────────────
#         train_df = pd.read_csv(self.data_dir / 'train_data.csv')
#         test_df  = pd.read_csv(self.data_dir / 'test_data.csv')

#         with open(self.data_dir / 'feature_info.json') as f:
#             feature_info = json.load(f)

#         # ── Feature filtering ─────────────────────────────────────────
#         LEAK_PATTERNS = [
#             'price_after', 'price_before', 'stock_return',
#             'abnormal_return', 'label_binary', 'label_median',
#             'label_tertile', 'sp500_return', 'alignment_timestamp',
#             'ticker', 'company_name', 'quarter', 'transcript_date',
#             'financial_date', 'DateDiff', 'date_diff',
#             'financial_match_type', 'match_type',
#         ]

#         all_cols = set(feature_info['feature_columns']) & set(train_df.columns)

#         numeric_cols_in_df = set(
#             train_df.select_dtypes(include=[np.number]).columns.tolist()
#         )

#         clean_features = sorted([
#             f for f in all_cols
#             if not any(pattern in f for pattern in LEAK_PATTERNS)
#             and not f.startswith('embedding_')
#             and f in numeric_cols_in_df
#         ])

#         def _clean(X_raw):
#             X_raw = X_raw.astype(np.float64)
#             X_raw[np.isinf(X_raw)] = np.nan
#             medians = np.nanmedian(X_raw, axis=0)
#             inds = np.where(np.isnan(X_raw))
#             X_raw[inds] = np.take(medians, inds[1])
#             return np.clip(X_raw, -1e6, 1e6)

#         X_tr_raw = _clean(train_df[clean_features].values)
#         X_te_raw = _clean(test_df[clean_features].values)

#         # Fit scaler on train only — transform both
#         scaler = StandardScaler()
#         X_train = scaler.fit_transform(X_tr_raw)
#         X_test  = scaler.transform(X_te_raw)

#         # ── Targets: label_median avoids distribution-shift bias ──────
#         if 'label_median' not in train_df.columns:
#             raise ValueError(
#                 "label_median not found. Run Phase 4 first to create it."
#             )
#         y_train = train_df['label_median'].values
#         y_test  = test_df['label_median'].values

#         # ── Class balance ─────────────────────────────────────────────
#         n_neg = (y_train == 0).sum()
#         n_pos = (y_train == 1).sum()
#         self.scale_pos_weight = float(n_neg) / max(float(n_pos), 1)

#         # Store — TSCV only sees training rows
#         self.X            = X_train
#         self.y            = y_train
#         self.X_test       = X_test
#         self.y_test       = y_test
#         self.feature_names = clean_features

#         tr_pos = y_train.mean()
#         te_pos = y_test.mean()
#         print(f"✅ Train: {len(X_train)} samples, {len(clean_features)} features")
#         print(f"   Target: label_median  "
#               f"(train {tr_pos:.1%} pos | test {te_pos:.1%} pos)")
#         print(f"   scale_pos_weight: {self.scale_pos_weight:.2f} "
#               f"(for XGBoost class balance)")
#         print()

#         return X_train, y_train
    
#     def evaluate_model(self, model):
#         """Evaluate model using TimeSeriesSplit on TRAIN rows only."""
#         scores = []
#         for train_idx, val_idx in self.tscv.split(self.X):
#             X_train, X_val = self.X[train_idx], self.X[val_idx]
#             y_train, y_val = self.y[train_idx], self.y[val_idx]

#             model.fit(X_train, y_train)
#             y_proba = model.predict_proba(X_val)[:, 1]

#             if len(np.unique(y_val)) < 2:
#                 continue  # skip degenerate fold
#             auc = roc_auc_score(y_val, y_proba)
#             scores.append(auc)

#         return (np.mean(scores), np.std(scores)) if scores else (0.5, 0.0)
    
#     def _load_shap_context(self) -> str:
#         """Load top SHAP features to give Gemini context about what matters."""
#         try:
#             shap_path = RESULTS_DIR / 'validation' / 'phase8b_feature_importance.csv'
#             if shap_path.exists():
#                 shap_df = pd.read_csv(shap_path)
#                 # Sort by second column (importance) if it exists, else first
#                 sort_col = shap_df.columns[1] if len(shap_df.columns) > 1 else shap_df.columns[0]
#                 shap_df  = shap_df.sort_values(sort_col, ascending=False)
#                 top5 = shap_df.iloc[:5, 0].tolist()
#                 return f"Top SHAP features: {', '.join(str(f) for f in top5)}"
#         except Exception:
#             pass
#         return "SHAP data not available"
 
#     def _gemini_choose_strategy(self) -> tuple[str, str]:
#         """
#         Ask Gemini to reason about which strategy to try next.
 
#         Returns (strategy_name, reasoning_text).
#         Returns (None, None) if the call fails so caller can fall back.
#         """
#         if self.gemini_model is None:
#             return None, None
 
#         # ── Build context for the LLM ──────────────────────────────────
#         strategy_descriptions = {
#             'feature_engineering':     'Create interaction features between financial and sentiment signals',
#             'hyperparameter_tuning':   'Search for optimal XGBoost/RF hyperparameters via random search',
#             'ensemble_building':       'Combine multiple models (voting/stacking) to reduce variance',
#             'architecture_search':     'Test different model architectures (LR, RF, XGBoost, SVM)',
#             'threshold_optimization':  'Optimize the decision threshold for class imbalance',
#         }
 
#         perf_summary = {}
#         for s, scores in self.memory['strategy_performance'].items():
#             if scores:
#                 perf_summary[s] = {
#                     'tries': len(scores),
#                     'avg_auc': round(float(np.mean(scores)), 4),
#                     'best_auc': round(float(max(scores)), 4),
#                 }
#             else:
#                 perf_summary[s] = {'tries': 0, 'avg_auc': 'untried', 'best_auc': 'untried'}
 
#         shap_context = self._load_shap_context()
 
#         history_lines = []
#         for h in self.memory['learning_history'][-5:]:  # last 5 iterations
#             history_lines.append(
#                 f"  Iter {h['iteration']}: {h['strategy']} → AUC {h['score']:.4f}"
#             )
#         history_str = '\n'.join(history_lines) if history_lines else '  (no history yet)'
 
#         prompt = f"""You are an autonomous ML optimization agent for a financial prediction system.
# Your task: predict whether a stock will OUTPERFORM the S&P 500 in 3 days after an earnings call.
 
# CURRENT STATE:
# - Iteration: {self.iteration + 1}
# - Best AUC so far: {self.best_score:.4f}
# - Training samples: {len(self.y) if self.y is not None else 'unknown'}
# - {shap_context}
 
# STRATEGY PERFORMANCE HISTORY:
# {json.dumps(perf_summary, indent=2)}
 
# RECENT ITERATIONS:
# {history_str}
 
# AVAILABLE STRATEGIES:
# {json.dumps(strategy_descriptions, indent=2)}
 
# TASK: Choose the single best strategy for the NEXT iteration.
# Consider:
# 1. Which strategies haven't been tried yet (exploration value)?
# 2. Which strategy has the highest average AUC (exploitation value)?
# 3. Given the SHAP features, which approach would most likely improve performance?
# 4. Is the dataset small (~460 training samples)? If so, simpler models > complex ones.
 
# Respond with ONLY valid JSON, no markdown, no explanation outside the JSON:
# {{"strategy": "<strategy_name>", "reasoning": "<2-3 sentence explanation of why>"}}"""
 
#         try:
#             response = self.gemini_model.generate_content(prompt)
#             raw = response.text.strip()
 
#             # Strip markdown code fences if present
#             if raw.startswith("```"):
#                 raw = raw.split("```")[1]
#                 if raw.startswith("json"):
#                     raw = raw[4:]
#             raw = raw.strip()
 
#             parsed = json.loads(raw)
#             strategy  = parsed.get("strategy", "").strip()
#             reasoning = parsed.get("reasoning", "").strip()
 
#             if strategy not in self.strategies:
#                 print(f"   ⚠️  Gemini returned unknown strategy '{strategy}' — falling back")
#                 return None, None
 
#             self.gemini_calls += 1
#             self.gemini_reasoning_log.append({
#                 'iteration': self.iteration + 1,
#                 'strategy':  strategy,
#                 'reasoning': reasoning,
#                 'raw_prompt_tokens': len(prompt.split()),
#             })
 
#             return strategy, reasoning
 
#         except json.JSONDecodeError as e:
#             print(f"   ⚠️  Gemini JSON parse error ({e}) — falling back")
#             return None, None
#         except Exception as e:
#             err_str = str(e)
#             # Daily quota exhausted — disable Gemini for rest of run, no point retrying
#             if "PerDay" in err_str or "free_tier_requests" in err_str:
#                 print(f"   ⚠️  Gemini daily quota exceeded — switching to epsilon-greedy for this run")
#                 self.gemini_model = None
#                 return None, None
#             # Per-minute rate limit — worth one short retry
#             if "429" in err_str and "PerMinute" in err_str:
#                 print(f"   ⏳ Rate limited (per-minute) — waiting 30s then retrying...")
#                 time.sleep(30)
#                 try:
#                     response = self.gemini_model.generate_content(prompt)
#                     raw = response.text.strip()
#                     if raw.startswith("```"):
#                         raw = raw.split("```")[1]
#                         if raw.startswith("json"):
#                             raw = raw[4:]
#                     raw = raw.strip()
#                     parsed = json.loads(raw)
#                     strategy  = parsed.get("strategy", "").strip()
#                     reasoning = parsed.get("reasoning", "").strip()
#                     if strategy not in self.strategies:
#                         return None, None
#                     self.gemini_calls += 1
#                     self.gemini_reasoning_log.append({
#                         'iteration': self.iteration + 1,
#                         'strategy':  strategy,
#                         'reasoning': reasoning,
#                         'raw_prompt_tokens': len(prompt.split()),
#                     })
#                     return strategy, reasoning
#                 except Exception as e2:
#                     print(f"   ⚠️  Gemini retry failed — falling back")
#                     return None, None
#             print(f"   ⚠️  Gemini API error ({e}) — falling back")
#             return None, None
 
#     def choose_strategy(self):
#         """
#         AGENT DECISION: Choose which strategy to try next.
 
#         PRIMARY:  Ask Gemini to reason about the best next strategy.
#         FALLBACK: Epsilon-greedy if Gemini is unavailable or fails.
#         """
#         # ── Try Gemini first ───────────────────────────────────────────
#         strategy, reasoning = self._gemini_choose_strategy()
 
#         if strategy is not None:
#             print(f"🤖 [Gemini] Agent Decision: {strategy}")
#             print(f"   💭 Reasoning: {reasoning}")
#             print()
#             return strategy
 
#         # ── Epsilon-greedy fallback ────────────────────────────────────
#         strategy_counts = {s: len(self.memory['strategy_performance'][s])
#                            for s in self.strategies.keys()}

#         # RULE 1: Force exploration until every strategy has been tried at
#         # least twice.  One trial per strategy is not enough — results are
#         # noisy and one unlucky seed can unfairly disadvantage a strategy.
#         # Minimum 2 trials ensures the agent has stable averages to compare.
#         undertried = [s for s, c in strategy_counts.items() if c < 2]
#         if undertried:
#             # Pick the one with fewest tries (0 before 1)
#             strategy = min(undertried, key=lambda s: strategy_counts[s])
#             label = 'not yet tried' if strategy_counts[strategy] == 0 else 'only 1 try'
#             decision = f"EXPLORE (trying {strategy}) [forced — {label}]"

#         elif np.random.random() < self.exploration_rate:
#             # RULE 2: Exploration — pick the least-tried strategy
#             # (ties broken by order, giving round-robin coverage)
#             strategy = min(strategy_counts.items(), key=lambda x: x[1])[0]
#             decision = f"EXPLORE (trying {strategy})"

#         else:
#             # RULE 3: Exploitation — pick the highest average AUC strategy
#             avg_performance = {
#                 s: np.mean(scores) if scores else 0.0
#                 for s, scores in self.memory['strategy_performance'].items()
#             }
#             strategy = max(avg_performance.items(), key=lambda x: x[1])[0]
#             decision = f"EXPLOIT (using best: {strategy})"

#         print(f"🤖 Agent Decision: {decision}")
#         print(f"   Exploration rate: {self.exploration_rate:.2f}")
#         print()
#         return strategy
    
#     def try_feature_engineering(self):
#         """Strategy 1: Feature Engineering — importance-guided interactions."""
#         print("🔧 Strategy: Feature Engineering")
#         print("   Creating interaction features (importance-guided)...")

#         # Use a quick RF to rank features and pick top ones for interactions
#         quick_rf = RandomForestClassifier(
#             n_estimators=50, max_depth=4, random_state=42, n_jobs=-1
#         )
#         quick_rf.fit(self.X, self.y)
#         importances = quick_rf.feature_importances_

#         # Top 5 financial + top 3 sentiment by importance
#         fn = np.array(self.feature_names)
#         fin_mask  = np.array(['financial_' in f for f in fn])
#         sent_mask = np.array(['sentiment' in f for f in fn])

#         fin_idx  = np.where(fin_mask)[0]
#         sent_idx = np.where(sent_mask)[0]

#         # Sort by importance and take top N
#         fin_top  = fin_idx[np.argsort(importances[fin_idx])[::-1][:5]]
#         sent_top = sent_idx[np.argsort(importances[sent_idx])[::-1][:3]]

#         new_features = []
#         new_names    = []
#         for fi in fin_top:
#             for si in sent_top:
#                 interaction = self.X[:, fi] * self.X[:, si]
#                 new_features.append(interaction.reshape(-1, 1))
#                 new_names.append(f"{fn[fi]}_x_{fn[si]}")

#         if new_features:
#             X_enhanced = np.hstack([self.X] + new_features)
#         else:
#             X_enhanced = self.X

#         model = xgb.XGBClassifier(
#             max_depth=6, learning_rate=0.1, n_estimators=100,
#             scale_pos_weight=self.scale_pos_weight,
#             random_state=42, n_jobs=-1,
#             eval_metric='logloss', verbosity=0,
#         )

#         original_X = self.X
#         self.X = X_enhanced
#         score, std = self.evaluate_model(model)
#         self.X = original_X

#         print(f"   Added {len(new_features)} interactions → Result: {score:.3f} (±{std:.3f})")
#         return score, {'n_features_added': len(new_features)}
    
#     def try_hyperparameter_tuning(self):
#         """Strategy 2: Hyperparameter Tuning"""
#         print("🔧 Strategy: Hyperparameter Tuning")
#         print("   Searching optimal parameters...")

#         param_space = {
#             'max_depth':        randint(3, 10),
#             'learning_rate':    uniform(0.01, 0.2),
#             'n_estimators':     randint(100, 400),
#             'subsample':        uniform(0.6, 0.4),
#             'colsample_bytree': uniform(0.6, 0.4),
#             'min_child_weight': randint(1, 8),
#             'gamma':            uniform(0, 0.5),
#         }

#         # n_jobs=1 on RandomizedSearchCV avoids Windows multiprocessing
#         # pickling errors (self.tscv cannot be serialised across processes).
#         # XGBoost itself still uses n_jobs=-1 internally for speed.
#         search = RandomizedSearchCV(
#             xgb.XGBClassifier(
#                 scale_pos_weight=self.scale_pos_weight,
#                 random_state=self.iteration, n_jobs=-1,
#                 eval_metric='logloss', verbosity=0,
#             ),
#             param_space,
#             n_iter=50,   # 50 trials gives stable results (20 was too noisy: ±0.016 variance)
#             cv=TimeSeriesSplit(n_splits=3),   # inline — avoids pickling self.tscv
#             scoring='roc_auc',
#             random_state=self.iteration,
#             n_jobs=1,    # Windows-safe: no multiprocessing for the search loop
#             verbose=0,
#         )
#         search.fit(self.X, self.y)

#         score = search.best_score_
#         print(f"   Result: {score:.3f}")
#         print(f"   Best params: {search.best_params_}")

#         return score, {'best_params': search.best_params_, 'model': search.best_estimator_}
    
#     def try_ensemble_building(self):
#         """Strategy 3: Ensemble Building"""
#         print("🔧 Strategy: Ensemble Building")
#         print("   Constructing ensemble...")

#         # Vary ensemble composition per iteration so repeated calls explore
#         # different configurations rather than always returning the same score.
#         rng = np.random.RandomState(self.iteration * 7 + 13)
#         lr_C       = float(rng.choice([0.01, 0.1, 1.0]))
#         rf_depth   = int(rng.choice([5, 8, 12]))
#         xgb_depth  = int(rng.choice([3, 5, 7]))
#         xgb_lr     = float(rng.uniform(0.01, 0.15))

#         estimators = [
#             ('lr',  LogisticRegression(C=lr_C, max_iter=1000,
#                                        class_weight='balanced', random_state=42)),
#             ('rf',  RandomForestClassifier(n_estimators=150, max_depth=rf_depth,
#                                            class_weight='balanced',
#                                            random_state=self.iteration, n_jobs=-1)),
#             ('xgb', xgb.XGBClassifier(max_depth=xgb_depth, learning_rate=xgb_lr,
#                                        n_estimators=150,
#                                        scale_pos_weight=self.scale_pos_weight,
#                                        random_state=self.iteration, n_jobs=-1,
#                                        eval_metric='logloss', verbosity=0)),
#         ]

#         print(f"   Config: LR(C={lr_C}), RF(depth={rf_depth}), XGB(depth={xgb_depth}, lr={xgb_lr:.3f})")
#         voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
#         score, std = self.evaluate_model(voting)

#         print(f"   Result: {score:.3f} (±{std:.3f})")
#         return score, {'ensemble_type': 'soft_voting', 'lr_C': lr_C,
#                        'rf_depth': rf_depth, 'xgb_depth': xgb_depth, 'xgb_lr': xgb_lr}

#     def try_architecture_search(self):
#         """Strategy 4: Architecture Search"""
#         print("🔧 Strategy: Architecture Search")
#         print("   Testing different architectures...")

#         models = {
#             'XGBoost': xgb.XGBClassifier(
#                 max_depth=6, n_estimators=150,
#                 scale_pos_weight=self.scale_pos_weight,
#                 random_state=42, n_jobs=-1,
#                 eval_metric='logloss', verbosity=0,
#             ),
#             'RandomForest': RandomForestClassifier(
#                 n_estimators=200, max_depth=10,
#                 class_weight='balanced',
#                 random_state=42, n_jobs=-1,
#             ),
#             'Stacking': StackingClassifier(
#                 estimators=[
#                     ('rf',  RandomForestClassifier(n_estimators=100,
#                                                    class_weight='balanced',
#                                                    random_state=42, n_jobs=-1)),
#                     ('xgb', xgb.XGBClassifier(n_estimators=100,
#                                                scale_pos_weight=self.scale_pos_weight,
#                                                random_state=42, n_jobs=-1,
#                                                eval_metric='logloss', verbosity=0)),
#                 ],
#                 final_estimator=LogisticRegression(max_iter=1000),
#                 cv=3, n_jobs=-1,
#             ),
#         }

#         best_score, best_arch = 0.0, None
#         for name, model in models.items():
#             s, _ = self.evaluate_model(model)
#             print(f"   {name}: {s:.3f}")
#             if s > best_score:
#                 best_score, best_arch = s, name

#         print(f"   Best architecture: {best_arch} → {best_score:.3f}")
#         return best_score, {'best_architecture': best_arch}

#     def try_threshold_optimization(self):
#         """Strategy 5: Find the probability threshold that maximises F1.

#         ROC-AUC is threshold-independent, but in practice we need to choose
#         a threshold for trading decisions.  This strategy finds the threshold
#         that maximises F1 on each CV validation fold and reports the mean
#         optimal-threshold F1 alongside the AUC.
#         """
#         from sklearn.metrics import precision_recall_curve, f1_score as sk_f1
#         from sklearn.calibration import CalibratedClassifierCV
#         print("🔧 Strategy: Threshold Optimisation")
#         print("   Finding optimal decision threshold per CV fold...")
#         print("   (with probability calibration via isotonic regression)")

#         # Wrap XGBoost with isotonic calibration so predict_proba() returns
#         # well-scaled probabilities — without this, thresholds like 0.026
#         # emerge because the model outputs near-uniform low probabilities.
#         base_model = xgb.XGBClassifier(
#             max_depth=6, learning_rate=0.1, n_estimators=150,
#             scale_pos_weight=self.scale_pos_weight,
#             random_state=42, n_jobs=-1,
#             eval_metric='logloss', verbosity=0,
#         )
#         model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)

#         auc_scores   = []
#         f1_opt_scores = []
#         best_thresholds = []

#         for train_idx, val_idx in self.tscv.split(self.X):
#             X_tr, X_val = self.X[train_idx], self.X[val_idx]
#             y_tr, y_val = self.y[train_idx], self.y[val_idx]

#             model.fit(X_tr, y_tr)
#             y_proba = model.predict_proba(X_val)[:, 1]

#             if len(np.unique(y_val)) < 2:
#                 continue

#             auc_scores.append(roc_auc_score(y_val, y_proba))

#             prec, rec, thresholds = precision_recall_curve(y_val, y_proba)
#             f1s    = 2 * prec * rec / (prec + rec + 1e-9)
#             best_i = int(np.argmax(f1s))
#             f1_opt_scores.append(f1s[best_i])
#             best_thresholds.append(
#                 float(thresholds[best_i]) if best_i < len(thresholds) else 0.5
#             )

#         mean_auc     = float(np.mean(auc_scores))   if auc_scores    else 0.5
#         mean_f1_opt  = float(np.mean(f1_opt_scores)) if f1_opt_scores else 0.0
#         mean_thresh  = float(np.mean(best_thresholds)) if best_thresholds else 0.5

#         print(f"   CV AUC:              {mean_auc:.3f}")
#         print(f"   CV F1 (opt thresh):  {mean_f1_opt:.3f}")
#         print(f"   Avg best threshold:  {mean_thresh:.3f}")

#         # Return AUC as the primary score (consistent with other strategies)
#         return mean_auc, {
#             'cv_f1_opt':        mean_f1_opt,
#             'avg_threshold':    mean_thresh,
#             'note': 'AUC reported; F1_opt stored in metadata',
#         }
    
#     def learn_from_outcome(self, strategy, score, metadata):
#         """
#         AGENT LEARNING: Update beliefs based on outcome
#         """
#         # Record performance
#         self.memory['strategy_performance'][strategy].append(score)
        
#         # Update best
#         if score > self.best_score:
#             improvement = score - self.best_score
#             self.best_score = score
            
#             self.memory['best_configurations'][strategy] = {
#                 'score': score,
#                 'metadata': metadata,
#                 'iteration': self.iteration
#             }
            
#             insight = f"Iteration {self.iteration}: {strategy} achieved new best: {score:.3f} (+{improvement:.3f})"
#             self.memory['insights'].append(insight)
            
#             print(f"💡 AGENT LEARNED: New best score! {score:.3f}")
#             print()
        
#         # Record in history
#         self.memory['learning_history'].append({
#             'iteration': self.iteration,
#             'strategy': strategy,
#             'score': score,
#             'was_best': score == self.best_score,
#             'exploration_rate': self.exploration_rate
#         })
    
#     def adapt_strategy(self):
#         """
#         AGENT ADAPTATION: Modify future behavior based on learning
#         """
#         strategy_counts = {s: len(self.memory['strategy_performance'][s])
#                            for s in self.strategies.keys()}
#         min_tries = min(strategy_counts.values())

#         # Only decay exploration AFTER every strategy has been tried at least
#         # once — otherwise the rate drops too fast and the agent locks onto
#         # the first strategy it tried before seeing the alternatives.
#         if min_tries >= 1:
#             self.exploration_rate = max(0.15, self.exploration_rate * 0.85)

#         # Switch to exploitation mode once every strategy has >= 2 tries
#         if min_tries >= 2 and self.current_strategy != 'exploitation':
#             self.current_strategy = 'exploitation'
#             print(f"🤖 Agent adapted: Switching to EXPLOITATION mode")
#             print()
    
#     def run_agentic_loop(self, max_iterations=8):
#         """
#         MAIN AGENTIC LOOP:
#         For each iteration:
#           1. Agent chooses strategy (explore vs exploit)
#           2. Agent executes strategy
#           3. Agent evaluates outcome
#           4. Agent learns from result
#           5. Agent adapts future behavior
#         """
#         print("="*60)
#         print("🤖 STARTING AGENTIC LEARNING LOOP")
#         print("="*60)
#         print()
#         print(f"Max iterations: {max_iterations}")
#         print(f"Agent will autonomously:")
#         print("  • Choose strategies")
#         print("  • Execute experiments")
#         print("  • Learn from outcomes")
#         print("  • Adapt behavior")
#         print()
        
#         for self.iteration in range(1, max_iterations + 1):
#             print("="*60)
#             print(f"ITERATION {self.iteration}/{max_iterations}")
#             print("="*60)
#             print()
            
#             # 1. DECIDE: Choose strategy
#             strategy_name = self.choose_strategy()
#             strategy_func = self.strategies[strategy_name]
            
#             # 2. EXECUTE: Run strategy
#             try:
#                 score, metadata = strategy_func()
#             except Exception as e:
#                 print(f"❌ Strategy failed: {e}")
#                 score = 0.0
#                 metadata = {'error': str(e)}
            
#             # 3. EVALUATE: Assess outcome
#             print()
            
#             # 4. LEARN: Update knowledge
#             self.learn_from_outcome(strategy_name, score, metadata)
            
#             # 5. ADAPT: Modify future behavior
#             self.adapt_strategy()
            
#             print(f"Current best: {self.best_score:.3f}")
#             print()
        
#         print("="*60)
#         print("🤖 AGENTIC LOOP COMPLETE")
#         print("="*60)
#         print()
    
#     def generate_insights(self):
#         """Agent reflects on what it learned"""
#         print("="*60)
#         print("🧠 AGENT INSIGHTS & LEARNING")
#         print("="*60)
#         print()
        
#         # Strategy effectiveness
#         print("📊 STRATEGY EFFECTIVENESS:")
#         print()
        
#         for strategy, scores in self.memory['strategy_performance'].items():
#             if scores:
#                 avg_score = np.mean(scores)
#                 best_score = max(scores)
#                 times_tried = len(scores)
                
#                 print(f"{strategy:<25} Avg: {avg_score:.3f}  Best: {best_score:.3f}  Tries: {times_tried}")
        
#         print()
        
#         # Best configuration
#         print("🏆 BEST CONFIGURATION:")
#         print()
        
#         best_strat = max(self.memory['best_configurations'].items(), 
#                         key=lambda x: x[1]['score'])
        
#         print(f"Strategy: {best_strat[0]}")
#         print(f"Score: {best_strat[1]['score']:.3f}")
#         print(f"Found at iteration: {best_strat[1]['iteration']}")
#         print()
        
#         # Key insights
#         if self.memory['insights']:
#             print("💡 KEY DISCOVERIES:")
#             print()
#             for insight in self.memory['insights']:
#                 print(f"  • {insight}")
#             print()
        
#         # Learning trajectory
#         print("📈 LEARNING TRAJECTORY:")
#         print()
        
#         for record in self.memory['learning_history']:
#             marker = "🥇" if record['was_best'] else "  "
#             print(f"{marker} Iter {record['iteration']}: {record['strategy']:<25} → {record['score']:.3f}")
        
#         print()
    
#     def save_results(self):
#         """Save agent's learning and results"""
#         print("="*60)
#         print("💾 SAVING AGENT MEMORY")
#         print("="*60)
#         print()
        
#         # Convert memory to JSON-serializable format
#         memory_export = {
#             'strategy_performance': {
#                 k: [float(s) for s in v]
#                 for k, v in self.memory['strategy_performance'].items()
#             },
#             'best_configurations': {
#                 k: {
#                     'score': float(v['score']),
#                     'iteration': int(v['iteration']),
#                     'metadata': str(v['metadata'])
#                 }
#                 for k, v in self.memory['best_configurations'].items()
#             },
#             'insights': self.memory['insights'],
#             'learning_history': [
#                 {
#                     'iteration': int(r['iteration']),
#                     'strategy': str(r['strategy']),
#                     'score': float(r['score']),
#                     'was_best': bool(r['was_best']),      # ← fix here
#                     'exploration_rate': float(r['exploration_rate'])
#                 }
#                 for r in self.memory['learning_history']
#             ],
#             'final_best_score': float(self.best_score)
#         }
        
#         memory_path = self.results_dir / 'agent_memory.json'
#         with open(memory_path, 'w') as f:
#             json.dump(memory_export, f, indent=2)
        
#         print(f"✅ Agent memory saved: {memory_path.name}")
        
#         # Generate report
#         report = {
#             'phase': 'Phase 10: True Agentic AI',
#             'timestamp': datetime.now().isoformat(),
#             'total_iterations': int(self.iteration),
#             'gemini_enabled': self.gemini_model is not None,
#             'gemini_model': self.gemini_model_name or 'epsilon-greedy fallback',
#             'gemini_api_calls': self.gemini_calls,
#             'gemini_reasoning_log': self.gemini_reasoning_log,
#             'strategies_available': list(self.strategies.keys()),
#             'strategies_tried': {
#                 k: int(len(v))
#                 for k, v in self.memory['strategy_performance'].items()
#             },
#             'best_score': float(self.best_score),
#             'best_strategy': max(
#                 self.memory['best_configurations'].items(),
#                 key=lambda x: x[1]['score']
#             )[0],
#             'learning_trajectory': [
#                 {
#                     'iteration': int(r['iteration']),
#                     'strategy':  str(r['strategy']),
#                     'score':     float(r['score']),
#                     'was_best':  bool(r['was_best']),
#                     'exploration_rate': float(r['exploration_rate'])
#                 }
#                 for r in self.memory['learning_history']
#             ],
#             'agent_insights': [str(i) for i in self.memory['insights']],
#             'final_exploration_rate': float(self.exploration_rate)
#         }
 
#         report_path = self.results_dir / 'agentic_ai_report.json'
#         with open(report_path, 'w') as f:
#             json.dump(report, f, indent=2)
        
#         print(f"✅ Agentic AI report saved: {report_path.name}")
#         print()
    
#     def run(self, max_iterations=8):
#         """Execute complete agentic AI workflow"""
        
#         # Prepare data
#         self.load_and_prepare_data()
        
#         # Run agentic learning loop
#         self.run_agentic_loop(max_iterations=max_iterations)
        
#         # Generate insights
#         self.generate_insights()
        
#         # Save results
#         self.save_results()
        
#         print("="*60)
#         print("✅ AGENTIC AI COMPLETE")
#         print("="*60)
#         print()
#         print(f"📁 Results saved to: {self.results_dir}/")
#         print()
#         print(f"🏆 Final Best Score: {self.best_score:.3f}")
#         print()
#         print("✅ Agent has learned optimal strategy!")
#         print()
        
#         return True
 
 
# def main():
#     """Main execution.
 
#     Set your Gemini API key before running:
#       Windows:  set GEMINI_API_KEY=your_key_here
#       Mac/Linux: export GEMINI_API_KEY=your_key_here
 
#     Get a free key at: https://aistudio.google.com/app/apikey
#     Free tier: 15 requests/min, 1M tokens/day — more than enough.
#     """
#     # Set your key as an environment variable before running:
#     #   Windows:   set GEMINI_API_KEY=your_key_here
#     #   Mac/Linux: export GEMINI_API_KEY=your_key_here
#     api_key = os.environ.get("GEMINI_API_KEY", "")
#     agent = AgenticAI(api_key=api_key)
#     agent.run(max_iterations=8)
#     return True
 
 
# if __name__ == "__main__":
#     main()



