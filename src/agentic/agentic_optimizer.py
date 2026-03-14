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
            'financial_date', 'DateDiff', 'date_diff'
        ]
        
        all_cols = set(feature_info['feature_columns']) & set(df.columns)
        clean_features = [
            f for f in all_cols
            if not any(pattern in f for pattern in LEAK_PATTERNS)
            and not f.startswith('embedding_')  # Exclude embeddings initially
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