"""
Phase 2C: Financial Data Normalization
Normalizes financial statement data for ML modeling
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from sklearn.preprocessing import StandardScaler, RobustScaler
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR

class FinancialNormalizer:
    """
    Normalizes financial statement data for machine learning
    """
    
    def __init__(self):
        self.input_dir = RAW_DATA_DIR / 'financial'
        self.output_dir = PROCESSED_DATA_DIR / 'financial'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'preprocessing' / 'financial_normalization.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Scalers for different feature types
        self.standard_scaler = StandardScaler()  # For normally distributed features
        self.robust_scaler = RobustScaler()      # For features with outliers
        
        self.normalization_stats = {
            'total_records': 0,
            'successfully_normalized': 0,
            'records_with_missing_values': 0,
            'features_normalized': 0,
            'outliers_detected': 0
        }
    
    def load_financial_features(self):
        """
        Load raw financial features from Phase 1C
        """
        features_file = self.input_dir / 'financial_features.csv'
        
        if not features_file.exists():
            print(f"❌ Financial features not found: {features_file}")
            print("   Please run Phase 1C first.")
            return None
        
        print(f"📂 Loading financial features from: {features_file}")
        df = pd.read_csv(features_file)
        
        print(f"✅ Loaded {len(df)} financial records")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print()
        
        return df
    
    def identify_feature_types(self, df):
        """
        Categorize financial features by type for appropriate normalization
        """
        # Exclude metadata columns
        metadata_cols = ['Ticker', 'Date']
        
        # All other columns are financial features
        financial_features = [col for col in df.columns if col not in metadata_cols]
        
        # Categorize by type
        # Income Statement features (tend to be normally distributed)
        income_features = [col for col in financial_features if any(
            keyword in col for keyword in ['Revenue', 'Income', 'Profit', 'EBIT', 'EPS', 'Tax']
        )]
        
        # Balance Sheet features (can have extreme values)
        balance_features = [col for col in financial_features if any(
            keyword in col for keyword in ['Assets', 'Liabilities', 'Equity', 'Debt', 'Cash', 'Working']
        )]
        
        # Cash Flow features (can have negative values and outliers)
        cashflow_features = [col for col in financial_features if any(
            keyword in col for keyword in ['Cash_Flow', 'Capital', 'Dividend', 'Repurchase']
        )]
        
        feature_categories = {
            'income_features': income_features,
            'balance_features': balance_features,
            'cashflow_features': cashflow_features,
            'all_features': financial_features
        }
        
        print("📊 FEATURE CATEGORIZATION:")
        print(f"   Income Statement features: {len(income_features)}")
        print(f"   Balance Sheet features: {len(balance_features)}")
        print(f"   Cash Flow features: {len(cashflow_features)}")
        print(f"   Total financial features: {len(financial_features)}")
        print()
        
        return feature_categories
    
    def handle_missing_values(self, df, feature_cols):
        """
        Handle missing values in financial data
        
        Strategy:
        1. Forward fill within same company (carry forward last known value)
        2. Fill remaining with industry median
        3. Fill any remaining with 0
        """
        df_filled = df.copy()
        
        missing_before = df_filled[feature_cols].isnull().sum().sum()
        
        print("🔧 HANDLING MISSING VALUES:")
        print(f"   Missing values before: {missing_before:,}")
        
        # Sort by Ticker and Date for forward fill
        df_filled = df_filled.sort_values(['Ticker', 'Date'])
        
        # Method 1: Forward fill within same ticker
        for col in feature_cols:
            df_filled[col] = df_filled.groupby('Ticker')[col].fillna(method='ffill')
        
        missing_after_ffill = df_filled[feature_cols].isnull().sum().sum()
        print(f"   After forward fill: {missing_after_ffill:,}")
        
        # Method 2: Fill with median
        for col in feature_cols:
            median_value = df_filled[col].median()
            df_filled[col].fillna(median_value, inplace=True)
        
        missing_after_median = df_filled[feature_cols].isnull().sum().sum()
        print(f"   After median fill: {missing_after_median:,}")
        
        # Method 3: Fill any remaining with 0
        df_filled[feature_cols] = df_filled[feature_cols].fillna(0)
        
        missing_final = df_filled[feature_cols].isnull().sum().sum()
        print(f"   Final missing values: {missing_final:,}")
        print()
        
        self.normalization_stats['records_with_missing_values'] = len(df) - len(df.dropna(subset=feature_cols))
        
        return df_filled
    
    def detect_outliers(self, df, feature_cols, threshold=3):
        """
        Detect outliers using Z-score method
        """
        outlier_counts = {}
        
        for col in feature_cols:
            # Calculate Z-scores
            mean = df[col].mean()
            std = df[col].std()
            
            if std > 0:
                z_scores = np.abs((df[col] - mean) / std)
                outliers = (z_scores > threshold).sum()
                outlier_counts[col] = outliers
        
        total_outliers = sum(outlier_counts.values())
        self.normalization_stats['outliers_detected'] = total_outliers
        
        print("🔍 OUTLIER DETECTION:")
        print(f"   Total outliers detected (Z > {threshold}): {total_outliers:,}")
        
        # Show top 5 features with most outliers
        if outlier_counts:
            top_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\n   Features with most outliers:")
            for feature, count in top_outliers:
                print(f"      {feature}: {count} outliers ({count/len(df)*100:.1f}%)")
        
        print()
        
        return outlier_counts
    
    def normalize_features(self, df, feature_categories):
        """
        Normalize features using appropriate scaling methods
        """
        df_normalized = df.copy()
        
        print("="*60)
        print("🔄 NORMALIZING FEATURES")
        print("="*60)
        print()
        
        # Get all financial features
        all_features = feature_categories['all_features']
        
        # Handle missing values first
        df_normalized = self.handle_missing_values(df_normalized, all_features)
        
        # Detect outliers
        outlier_counts = self.detect_outliers(df_normalized, all_features)
        
        # Normalize features
        print("📊 APPLYING NORMALIZATION:")
        
        # Use RobustScaler (resistant to outliers)
        # This scales features using median and IQR instead of mean and std
        print(f"   Using RobustScaler for all {len(all_features)} features")
        print(f"   Method: (X - median) / IQR")
        print()
        
        # Fit and transform
        feature_values = df_normalized[all_features].values
        normalized_values = self.robust_scaler.fit_transform(feature_values)
        
        # Replace original values with normalized values
        df_normalized[all_features] = normalized_values
        
        # Statistics
        print("📈 NORMALIZATION STATISTICS:")
        for i, feature in enumerate(all_features[:5]):  # Show first 5
            original_mean = df[feature].mean()
            original_std = df[feature].std()
            normalized_mean = df_normalized[feature].mean()
            normalized_std = df_normalized[feature].std()
            
            print(f"\n   {feature}:")
            print(f"      Original: μ={original_mean:.2e}, σ={original_std:.2e}")
            print(f"      Normalized: μ={normalized_mean:.3f}, σ={normalized_std:.3f}")
        
        print(f"\n   ... and {len(all_features) - 5} more features")
        print()
        
        self.normalization_stats['features_normalized'] = len(all_features)
        
        return df_normalized, all_features
    
    def create_derived_ratios(self, df):
        """
        Create financial ratios from normalized features
        These are standard ratios used in fundamental analysis
        """
        df_ratios = df.copy()
        
        print("📊 CREATING FINANCIAL RATIOS:")
        
        derived_features = []
        
        # Profitability Ratios
        if 'Net_Income' in df.columns and 'Total_Revenue' in df.columns:
            df_ratios['Profit_Margin'] = np.where(
                df['Total_Revenue'] != 0,
                df['Net_Income'] / df['Total_Revenue'],
                0
            )
            derived_features.append('Profit_Margin')
            print("   ✅ Profit_Margin = Net_Income / Total_Revenue")
        
        # Liquidity Ratios
        if 'Current_Assets' in df.columns and 'Current_Liabilities' in df.columns:
            df_ratios['Current_Ratio'] = np.where(
                df['Current_Liabilities'] != 0,
                df['Current_Assets'] / df['Current_Liabilities'],
                0
            )
            derived_features.append('Current_Ratio')
            print("   ✅ Current_Ratio = Current_Assets / Current_Liabilities")
        
        # Leverage Ratios
        if 'Total_Debt' in df.columns and 'Total_Equity' in df.columns:
            df_ratios['Debt_to_Equity'] = np.where(
                df['Total_Equity'] != 0,
                df['Total_Debt'] / df['Total_Equity'],
                0
            )
            derived_features.append('Debt_to_Equity')
            print("   ✅ Debt_to_Equity = Total_Debt / Total_Equity")
        
        # Efficiency Ratios
        if 'Total_Revenue' in df.columns and 'Total_Assets' in df.columns:
            df_ratios['Asset_Turnover'] = np.where(
                df['Total_Assets'] != 0,
                df['Total_Revenue'] / df['Total_Assets'],
                0
            )
            derived_features.append('Asset_Turnover')
            print("   ✅ Asset_Turnover = Total_Revenue / Total_Assets")
        
        # Return Ratios
        if 'Net_Income' in df.columns and 'Total_Assets' in df.columns:
            df_ratios['ROA'] = np.where(
                df['Total_Assets'] != 0,
                df['Net_Income'] / df['Total_Assets'],
                0
            )
            derived_features.append('ROA')
            print("   ✅ ROA = Net_Income / Total_Assets")
        
        if 'Net_Income' in df.columns and 'Total_Equity' in df.columns:
            df_ratios['ROE'] = np.where(
                df['Total_Equity'] != 0,
                df['Net_Income'] / df['Total_Equity'],
                0
            )
            derived_features.append('ROE')
            print("   ✅ ROE = Net_Income / Total_Equity")
        
        print(f"\n   Created {len(derived_features)} financial ratios")
        print()
        
        return df_ratios, derived_features
    
    def save_normalized_data(self, df_normalized, feature_cols, derived_features):
        """
        Save normalized financial data
        """
        try:
            print("="*60)
            print("💾 SAVING NORMALIZED DATA")
            print("="*60)
            print()
            
            # Save normalized features
            normalized_path = self.output_dir / 'financial_features_normalized.csv'
            df_normalized.to_csv(normalized_path, index=False)
            print(f"✅ Normalized features saved: {normalized_path}")
            print(f"   Size: {normalized_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save scaler for future use
            import joblib
            scaler_path = self.output_dir / 'robust_scaler.pkl'
            joblib.dump(self.robust_scaler, scaler_path)
            print(f"✅ Scaler saved: {scaler_path}")
            
            # Save feature metadata
            metadata = {
                'original_features': feature_cols,
                'derived_features': derived_features,
                'total_features': len(feature_cols) + len(derived_features),
                'scaler_type': 'RobustScaler',
                'normalization_date': datetime.now().isoformat()
            }
            
            metadata_path = self.output_dir / 'normalization_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Metadata saved: {metadata_path}")
            
            # Save statistics
            stats_to_save = {
                key: int(value) if isinstance(value, (np.int64, np.int32)) else value
                for key, value in self.normalization_stats.items()
            }

            stats_path = self.output_dir / 'normalization_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(stats_to_save, f, indent=2)
            print(f"✅ Statistics saved: {stats_path}")
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Error saving normalized data: {e}")
            return False
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self):
        """
        Execute complete financial normalization workflow
        """
        print("="*60)
        print("PHASE 2C: FINANCIAL DATA NORMALIZATION")
        print("="*60)
        print()
        
        # Load financial features
        df = self.load_financial_features()
        
        if df is None:
            print("❌ Normalization failed")
            return False
        
        self.normalization_stats['total_records'] = len(df)
        
        # Identify feature types
        feature_categories = self.identify_feature_types(df)
        
        # ── FIX: create ratios from RAW values first, THEN normalize ──
        # Previously ratios were computed from already-scaled values,
        # making them economically meaningless (e.g. scaled_NI / scaled_Revenue).
        # Now we compute them on the original dollar values so they retain
        # their real meaning (Profit_Margin = 0.25 means 25% margin).
        df_with_ratios, derived_features = self.create_derived_ratios(df)

        # Build updated feature categories that include the new ratio columns
        feature_categories['all_features'] = (
            feature_categories['all_features'] + derived_features
        )

        # Now normalize everything (original features + ratios) together
        df_normalized, feature_cols = self.normalize_features(
            df_with_ratios, feature_categories
        )
        
        self.normalization_stats['successfully_normalized'] = len(df_normalized)
        
        # Display summary
        print("="*60)
        print("📊 NORMALIZATION SUMMARY")
        print("="*60)
        print()
        print(f"Total records:           {self.normalization_stats['total_records']:,}")
        print(f"Successfully normalized: {self.normalization_stats['successfully_normalized']:,}")
        print(f"Features normalized:     {self.normalization_stats['features_normalized']}")
        print(f"Derived features:        {len(derived_features)}")
        print(f"Total features:          {self.normalization_stats['features_normalized']}")
        print()
        print(f"Records with missing:    {self.normalization_stats['records_with_missing_values']}")
        print(f"Outliers detected:       {self.normalization_stats['outliers_detected']:,}")
        print()
        
        # Save normalized data (derived_features=[] because already in feature_cols)
        success = self.save_normalized_data(df_normalized, feature_cols, [])
        
        if success:
            print("="*60)
            print("✅ PHASE 2C COMPLETE")
            print("="*60)
            print()
            print(f"📁 Normalized data saved to: {self.output_dir}")
            print(f"📝 Log saved to: {self.log_file}")
            print()
            print("✅ Ready for Phase 2D: Temporal Alignment")
            print()
        
        return success

def main():
    """Main execution"""
    normalizer = FinancialNormalizer()
    success = normalizer.run()
    return success

if __name__ == "__main__":
    main()