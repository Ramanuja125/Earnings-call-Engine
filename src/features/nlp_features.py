"""
Phase 3B: Additional NLP Features
Extracts Loughran-McDonald dictionary features and text statistics
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import PROCESSED_DATA_DIR, FEATURES_DIR, LOGS_DIR

class NLPFeatureExtractor:
    """
    Extracts NLP features using Loughran-McDonald dictionary and text statistics
    """
    
    def __init__(self):
        self.transcripts_dir = PROCESSED_DATA_DIR / 'transcripts'
        self.aligned_dir = PROCESSED_DATA_DIR / 'aligned'
        self.output_dir = FEATURES_DIR
        
        self.log_file = LOGS_DIR / 'features' / 'nlp_features.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Loughran-McDonald Dictionary (simplified version)
        # Full dictionary has thousands of words, we'll use key words
        self.lm_dict = {
            'negative': [
                'loss', 'losses', 'decline', 'declined', 'declining', 'decrease', 'decreased',
                'decreasing', 'risk', 'risks', 'risky', 'uncertain', 'uncertainty', 'challenge',
                'challenges', 'difficult', 'difficulty', 'weak', 'weakness', 'poor', 'negative',
                'negatively', 'adverse', 'adversely', 'concern', 'concerns', 'problem', 'problems',
                'threat', 'threats', 'volatile', 'volatility', 'downturn', 'recession', 'crisis',
                'fail', 'failure', 'failed', 'failing', 'litigation', 'lawsuit', 'default',
                'bankruptcy', 'restructuring', 'impairment', 'writedown', 'layoff', 'layoffs'
            ],
            'positive': [
                'gain', 'gains', 'growth', 'growing', 'grow', 'increase', 'increased', 'increasing',
                'profit', 'profits', 'profitable', 'profitability', 'strong', 'strength', 'improve',
                'improved', 'improvement', 'improving', 'opportunity', 'opportunities', 'success',
                'successful', 'achieve', 'achieved', 'achievement', 'positive', 'positively',
                'excellent', 'outstanding', 'robust', 'innovation', 'innovative', 'leader',
                'leadership', 'competitive', 'advantage', 'benefit', 'benefits', 'win', 'winning'
            ],
            'uncertainty': [
                'uncertain', 'uncertainty', 'uncertainties', 'risk', 'risks', 'risky', 'maybe',
                'perhaps', 'possibly', 'might', 'could', 'may', 'approximate', 'approximately',
                'depend', 'depends', 'depending', 'tentative', 'preliminary', 'subject to',
                'conditional', 'estimate', 'estimated', 'expect', 'expected', 'anticipate',
                'anticipated', 'believe', 'believes', 'appears', 'seems', 'likely', 'unlikely'
            ],
            'litigious': [
                'lawsuit', 'lawsuits', 'litigation', 'litigations', 'court', 'judge', 'jury',
                'trial', 'settlement', 'sue', 'sued', 'suing', 'legal', 'attorney', 'claim',
                'claims', 'complaint', 'complaints', 'allege', 'alleged', 'allegations'
            ],
            'constraining': [
                'must', 'shall', 'required', 'requirement', 'requirements', 'obligation',
                'obligations', 'obligated', 'mandate', 'mandated', 'mandatory', 'need', 'needs',
                'necessary', 'compel', 'compelled', 'force', 'forced', 'prohibit', 'prohibited'
            ]
        }
        
        self.extraction_stats = {
            'total_transcripts': 0,
            'successfully_extracted': 0,
            'failed': 0
        }
    
    def load_aligned_transcripts(self):
        """
        Load transcripts matching aligned data
        """
        print("="*60)
        print("PHASE 3B: ADDITIONAL NLP FEATURES")
        print("="*60)
        print()
        print("📂 Loading aligned transcripts...")
        
        # Load aligned data
        aligned_file = self.aligned_dir / 'aligned_data.csv'
        if not aligned_file.exists():
            print(f"❌ Aligned data not found: {aligned_file}")
            return None
        
        aligned_df = pd.read_csv(aligned_file)
        aligned_keys = set(zip(aligned_df['ticker'], aligned_df['quarter']))
        
        # Load transcripts
        transcripts_file = self.transcripts_dir / 'transcripts_segmented.json'
        if not transcripts_file.exists():
            print(f"❌ Transcripts not found: {transcripts_file}")
            return None
        
        with open(transcripts_file, 'r', encoding='utf-8') as f:
            all_transcripts = json.load(f)
        
        # Filter to aligned
        transcripts = [
            t for t in all_transcripts 
            if (t['ticker'], t['quarter']) in aligned_keys
        ]
        
        print(f"✅ Loaded {len(transcripts)} transcripts")
        print()
        
        self.extraction_stats['total_transcripts'] = len(transcripts)
        
        return transcripts
    
    def tokenize(self, text):
        """
        Simple tokenization - split on whitespace and remove punctuation
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation but keep words
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        return words
    
    def calculate_lm_scores(self, text):
        """
        Calculate Loughran-McDonald dictionary scores
        """
        words = self.tokenize(text)
        word_count = len(words)
        
        if word_count == 0:
            return {
                'lm_negative': 0,
                'lm_positive': 0,
                'lm_uncertainty': 0,
                'lm_litigious': 0,
                'lm_constraining': 0,
                'lm_negative_count': 0,
                'lm_positive_count': 0,
                'lm_uncertainty_count': 0,
                'lm_net_sentiment': 0
            }
        
        # Count words in each category
        counts = {}
        for category, word_list in self.lm_dict.items():
            word_set = set(word_list)
            count = sum(1 for word in words if word in word_set)
            counts[category] = count
        
        # Calculate proportions
        scores = {
            'lm_negative': counts['negative'] / word_count,
            'lm_positive': counts['positive'] / word_count,
            'lm_uncertainty': counts['uncertainty'] / word_count,
            'lm_litigious': counts['litigious'] / word_count,
            'lm_constraining': counts['constraining'] / word_count,
            'lm_negative_count': counts['negative'],
            'lm_positive_count': counts['positive'],
            'lm_uncertainty_count': counts['uncertainty'],
            'lm_net_sentiment': (counts['positive'] - counts['negative']) / word_count
        }
        
        return scores
    
    def calculate_text_statistics(self, text):
        """
        Calculate basic text statistics
        """
        # Word count
        words = self.tokenize(text)
        word_count = len(words)
        
        # Character count
        char_count = len(text)
        
        # Sentence count (rough estimate)
        sentences = text.split('.')
        sentence_count = len([s for s in sentences if s.strip()])
        
        # Average word length
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Unique words
        unique_words = len(set(words))
        lexical_diversity = unique_words / word_count if word_count > 0 else 0
        
        # Question count (rough estimate)
        question_count = text.count('?')
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': sentence_count,
            'avg_word_length': avg_word_length,
            'avg_sentence_length': avg_sentence_length,
            'unique_words': unique_words,
            'lexical_diversity': lexical_diversity,
            'question_count': question_count
        }
    
    def extract_features_from_transcript(self, transcript):
        """
        Extract all NLP features from a single transcript
        """
        try:
            # ── FIX: speaker_segmenter.py saves 'management_text' and
            # 'analyst_text' (no 'total_' prefix). The old code was looking
            # for 'total_management_text' which was always empty, causing
            # mgmt/analyst LM scores to be identical and divergence = 0.
            management_text = (
                transcript.get('management_text', '')
                or transcript.get('total_management_text', '')   # legacy fallback
                or ''
            )
            analyst_text = (
                transcript.get('analyst_text', '')
                or transcript.get('total_analyst_text', '')      # legacy fallback
                or ''
            )

            # If both sections are still empty, fall back to prepared_remarks / QA
            if not management_text.strip() and not analyst_text.strip():
                full = (
                    transcript.get('full_text_cleaned', '')
                    or transcript.get('full_text', '')
                    or transcript.get('prepared_remarks_cleaned', '')
                    or transcript.get('prepared_remarks', '')
                    or ''
                )
                # Try to split at Q&A markers
                split_markers = ['QUESTION AND ANSWER', 'Q&A', 'QUESTIONS AND ANSWERS',
                                 'Question-and-Answer', 'Questions and Answers']
                split_idx = len(full)
                for marker in split_markers:
                    idx = full.upper().find(marker.upper())
                    if idx != -1 and idx < split_idx:
                        split_idx = idx
                management_text = full[:split_idx]
                analyst_text    = full[split_idx:]
                full_text       = full
            else:
                full_text = management_text + ' ' + analyst_text
            
            # Extract LM scores
            full_lm = self.calculate_lm_scores(full_text)
            mgmt_lm = self.calculate_lm_scores(management_text)
            analyst_lm = self.calculate_lm_scores(analyst_text)
            
            # Extract text statistics
            full_stats = self.calculate_text_statistics(full_text)
            mgmt_stats = self.calculate_text_statistics(management_text)
            analyst_stats = self.calculate_text_statistics(analyst_text)
            
            # Combine features
            features = {
                # Metadata
                'ticker': transcript['ticker'],
                'quarter': transcript.get('quarter', ''),
                'date': transcript.get('date', ''),
                
                # Full transcript LM scores
                **{k: v for k, v in full_lm.items()},
                
                # Management LM scores (prefix with mgmt_)
                **{f'mgmt_{k}': v for k, v in mgmt_lm.items()},
                
                # Analyst LM scores (prefix with analyst_)
                **{f'analyst_{k}': v for k, v in analyst_lm.items()},
                
                # Full transcript text statistics
                **{k: v for k, v in full_stats.items()},
                
                # Management text statistics
                'mgmt_word_count': mgmt_stats['word_count'],
                'mgmt_sentence_count': mgmt_stats['sentence_count'],
                'mgmt_avg_word_length': mgmt_stats['avg_word_length'],
                
                # Analyst text statistics
                'analyst_word_count': analyst_stats['word_count'],
                'analyst_sentence_count': analyst_stats['sentence_count'],
                'analyst_avg_word_length': analyst_stats['avg_word_length'],
                
                # Divergence metrics
                'lm_sentiment_divergence': mgmt_lm['lm_net_sentiment'] - analyst_lm['lm_net_sentiment'],
                'uncertainty_divergence': mgmt_lm['lm_uncertainty'] - analyst_lm['lm_uncertainty']
            }
            
            self.extraction_stats['successfully_extracted'] += 1
            
            return features
            
        except Exception as e:
            self._log_error(f"Error extracting features from {transcript.get('ticker', 'UNKNOWN')}: {e}")
            self.extraction_stats['failed'] += 1
            return None
    
    def extract_all_features(self, transcripts):
        """
        Extract features from all transcripts
        """
        print("="*60)
        print("📊 EXTRACTING NLP FEATURES")
        print("="*60)
        print()
        print("🔤 Processing Loughran-McDonald dictionary...")
        print("📈 Calculating text statistics...")
        print()
        
        all_features = []
        used_segmented = 0   # how many used the proper mgmt/analyst split
        used_fallback  = 0   # how many fell back to full-text splitting

        from tqdm import tqdm
        for transcript in tqdm(transcripts, desc="Processing"):
            # Track whether proper segmentation data is available
            has_mgmt = bool(
                transcript.get('management_text', '').strip()
                or transcript.get('total_management_text', '').strip()
            )
            features = self.extract_features_from_transcript(transcript)
            if features:
                all_features.append(features)
                if has_mgmt:
                    used_segmented += 1
                else:
                    used_fallback += 1
        
        print()
        print("="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        print()
        print(f"Total transcripts:       {self.extraction_stats['total_transcripts']}")
        print(f"Successfully extracted:  {self.extraction_stats['successfully_extracted']}")
        print(f"Failed:                  {self.extraction_stats['failed']}")
        print()
        print(f"📊 Speaker Separation Quality:")
        print(f"   Used proper mgmt/analyst split: {used_segmented}")
        print(f"   Used full-text fallback:        {used_fallback}")
        if used_fallback > 0:
            print(f"   ⚠️  {used_fallback} transcripts had no speaker segmentation data.")
            print(f"      Check that Phase 2B saved 'management_text' and 'analyst_text' fields.")
        
        if self.extraction_stats['successfully_extracted'] > 0:
            success_rate = self.extraction_stats['successfully_extracted'] / self.extraction_stats['total_transcripts']
            print(f"   Success rate: {success_rate:.1%}")
        
        print()
        
        return all_features
    
    def save_features(self, features):
        """
        Save extracted NLP features
        """
        if not features:
            print("❌ No features to save")
            return False
        
        try:
            print("="*60)
            print("💾 SAVING NLP FEATURES")
            print("="*60)
            print()
            
            # Convert to DataFrame
            features_df = pd.DataFrame(features)
            
            # Save as CSV
            csv_path = self.output_dir / 'nlp_features.csv'
            features_df.to_csv(csv_path, index=False)
            print(f"✅ Features saved: {csv_path}")
            print(f"   Size: {csv_path.stat().st_size / (1024):.2f} KB")
            print(f"   Records: {len(features_df):,}")
            print(f"   Features: {len(features_df.columns)}")
            
            # Save statistics
            stats_path = self.output_dir / 'nlp_extraction_stats.json'
            with open(stats_path, 'w') as f:
                json.dump(self.extraction_stats, f, indent=2)
            print(f"✅ Statistics saved: {stats_path}")
            
            # Display summary statistics
            print()
            print("📊 FEATURE SUMMARY:")
            print(f"   Loughran-McDonald features: 27")
            print(f"   Text statistics: 17")
            print(f"   Total NLP features: {len(features_df.columns) - 3}")  # Minus metadata
            print()
            
            print("📈 KEY METRICS:")
            print(f"   Average uncertainty: {features_df['lm_uncertainty'].mean():.4f}")
            print(f"   Average negative tone: {features_df['lm_negative'].mean():.4f}")
            print(f"   Average positive tone: {features_df['lm_positive'].mean():.4f}")
            print(f"   Average word count: {features_df['word_count'].mean():.0f}")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving features: {e}")
            return False
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self):
        """
        Execute complete NLP feature extraction workflow
        """
        # Load transcripts
        transcripts = self.load_aligned_transcripts()
        
        if transcripts is None:
            print("❌ NLP feature extraction failed")
            return False
        
        # Extract features
        features = self.extract_all_features(transcripts)
        
        # Save features
        success = self.save_features(features)
        
        if success:
            print("="*60)
            print("✅ PHASE 3B COMPLETE")
            print("="*60)
            print()
            print(f"📁 Features saved to: {self.output_dir}")
            print(f"📝 Log saved to: {self.log_file}")
            print()
            print("✅ Ready for Phase 3C: Feature Integration")
            print()
        
        return success

def main():
    """Main execution"""
    extractor = NLPFeatureExtractor()
    success = extractor.run()
    return success

if __name__ == "__main__":
    main()