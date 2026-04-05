"""
Phase 2A: Transcript Cleaning and Normalization
Cleans earnings call transcripts for NLP processing
"""

import pandas as pd
import json
import re
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, LOGS_DIR

class TranscriptCleaner:
    """
    Cleans and normalizes earnings call transcripts
    """
    
    def __init__(self):
        self.input_dir = RAW_DATA_DIR / 'transcripts'
        self.output_dir = PROCESSED_DATA_DIR / 'transcripts'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = LOGS_DIR / 'preprocessing' / 'transcript_cleaning.log'
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.cleaning_stats = {
            'total_processed': 0,
            'successfully_cleaned': 0,
            'failed': 0,
            'avg_original_length': 0,
            'avg_cleaned_length': 0,
            'total_chars_removed': 0
        }
    
    def remove_boilerplate(self, text):
        """
        Remove earnings call boilerplate that contaminates sentiment analysis.
        Targets: safe harbor statements, greetings, operator instructions.
        These are uniformly positive/neutral formal language that drowns
        out actual financial sentiment signal.
        """
        if not text or not isinstance(text, str):
            return text

        # Safe harbor / forward-looking statements
        import re
        patterns = [
            # Safe harbor blocks
            r'(?si)this (?:call|conference|presentation|release|discussion) (?:contains?|includes?|may contain) forward.{0,20}looking statements?.{0,600}(?=\n\n|\Z)',
            r'(?si)(?:caution|cautionary).{0,30}forward.{0,10}looking.{0,400}(?=\n\n|\Z)',
            r'(?si)actual results? (?:may|could|might) (?:differ|vary).{0,400}(?=\n\n|\Z)',
            r'(?si)(?:annual|quarterly) report on form \d+.{0,200}(?=\n\n|\Z)',
            # Opening greetings (only match near start of text)
            r'(?i)^.{0,400}(?:good (?:morning|afternoon|evening)[,.]?\s+(?:everyone|ladies|and welcome)|welcome to the .{0,60}(?:earnings|results|quarter).{0,60}(?:call|conference)|thank you for (?:joining|standing by|participating))[^\n]{0,200}',
            # Operator boilerplate
            r'(?im)^operator\s*:?.{0,200}(?:press \*|mute|queue|questions? and answers?|Q&A session)[^\n]{0,200}$',
            r'(?i)(?:ladies and gentlemen,? )?(?:this concludes|thank you for joining|you may (?:now )?disconnect)[^\n]{0,200}',
        ]
        for pat in patterns:
            text = re.sub(pat, ' ', text)

        # Clean up extra whitespace
        text = re.sub(r'  +', ' ', text).strip()
        return text

    def normalize_text(self, text):
        """
        Normalize transcript text
        
        Steps:
        1. Convert to lowercase (optional - preserve case for NER)
        2. Remove excessive whitespace
        3. Remove special characters (preserve sentence structure)
        4. Normalize punctuation
        5. Remove filler words/sounds
        """
        if not text or not isinstance(text, str):
            return ""
        
        original_length = len(text)
        
        # Step 1: Normalize whitespace (multiple spaces/newlines to single space)
        text = re.sub(r'\s+', ' ', text)
        
        # Step 2: Remove common transcript artifacts
        # Remove [pause], [music], [background noise], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        # Remove HTML/XML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Step 3: Normalize punctuation
        # Fix multiple periods, exclamations, questions
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        # Step 4: Remove filler words/sounds (optional - keep for sentiment)
        # Common fillers: um, uh, er, ah, you know, etc.
        # SKIP THIS - fillers can indicate uncertainty which is valuable
        
        # Step 5: Remove excessive punctuation but preserve sentence structure
        # Keep: . , ! ? : ; - ( )
        # Remove: @ # $ % ^ & * ~ ` 
        text = re.sub(r'[@#$%^&*~`]', '', text)
        
        # Step 6: Normalize dashes and hyphens
        text = re.sub(r'—|–', '-', text)
        text = re.sub(r'\s+-\s+', ' - ', text)  # Normalize spacing around dashes
        
        # Step 7: Fix common OCR/transcription errors
        # Fix multiple spaces after punctuation
        text = re.sub(r'([.!?])\s+', r'\1 ', text)
        
        # Step 8: Strip leading/trailing whitespace
        text = text.strip()
        
        # Calculate stats
        cleaned_length = len(text)
        chars_removed = original_length - cleaned_length
        
        return text, chars_removed
    
    def extract_metadata(self, transcript_dict):
        """
        Extract and validate metadata from transcript
        """
        metadata = {
            'ticker': transcript_dict.get('ticker', ''),
            'company_name': transcript_dict.get('company_name', ''),
            'quarter': transcript_dict.get('quarter', ''),
            'date': transcript_dict.get('date', ''),
            'fiscal_year': transcript_dict.get('fiscal_year', 0),
            'fiscal_quarter': transcript_dict.get('fiscal_quarter', 0),
            'source': transcript_dict.get('source', 'UNKNOWN')
        }
        
        return metadata
    
    def clean_section(self, section_text):
        """
        Clean a specific section (prepared remarks or Q&A)
        """
        if not section_text or not isinstance(section_text, str):
            return "", 0
        
        cleaned_text, chars_removed = self.normalize_text(section_text)
        
        return cleaned_text, chars_removed
    
    def clean_transcript(self, transcript_dict):
        """
        Clean a single transcript
        """
        try:
            # Extract metadata
            metadata = self.extract_metadata(transcript_dict)
            
            # Get original text sections
            prepared_remarks = transcript_dict.get('prepared_remarks', '')
            qa_section = transcript_dict.get('qa_section', '')
            full_text = transcript_dict.get('full_text', '')

            # Strip boilerplate before NLP — safe harbor/greetings bias sentiment
            prepared_remarks = self.remove_boilerplate(prepared_remarks)
            qa_section       = self.remove_boilerplate(qa_section)
            full_text        = self.remove_boilerplate(full_text)

            # Clean each section
            cleaned_prepared, chars_removed_prep = self.clean_section(prepared_remarks)
            cleaned_qa, chars_removed_qa = self.clean_section(qa_section)
            cleaned_full, chars_removed_full = self.clean_section(full_text)
            
            # Create cleaned transcript dictionary
            cleaned_transcript = {
                **metadata,
                'prepared_remarks_original_length': len(prepared_remarks),
                'prepared_remarks_cleaned_length': len(cleaned_prepared),
                'prepared_remarks_cleaned': cleaned_prepared,
                
                'qa_section_original_length': len(qa_section),
                'qa_section_cleaned_length': len(cleaned_qa),
                'qa_section_cleaned': cleaned_qa,
                
                'full_text_original_length': len(full_text),
                'full_text_cleaned_length': len(cleaned_full),
                'full_text_cleaned': cleaned_full,
                
                'total_chars_removed': chars_removed_prep + chars_removed_qa,
                'cleaning_timestamp': datetime.now().isoformat()
            }
            
            # Update stats
            self.cleaning_stats['total_chars_removed'] += cleaned_transcript['total_chars_removed']
            
            return cleaned_transcript
            
        except Exception as e:
            self._log_error(f"Error cleaning transcript {metadata.get('ticker', 'UNKNOWN')}: {e}")
            return None
    
    def clean_all_transcripts(self, real_only=False):  # ← Add parameter
        """
        Clean all transcripts from the raw data
        
        Args:
            real_only: If True, only process real transcripts. If False, include synthetic.
        """
        print("="*60)
        print("PHASE 2A: TRANSCRIPT CLEANING & NORMALIZATION")
        print("="*60)
        print()
        
        # Load raw transcripts
        transcripts_file = self.input_dir / 'transcripts_full.json'
        
        if not transcripts_file.exists():
            print(f"❌ Transcript file not found: {transcripts_file}")
            return None
        
        print(f"📂 Loading transcripts from: {transcripts_file}")
        
        with open(transcripts_file, 'r', encoding='utf-8') as f:
            raw_transcripts = json.load(f)
        
        print(f"✅ Loaded {len(raw_transcripts)} raw transcripts")
        print()
        
        # Filter: real transcripts only (CONFIGURABLE)
        if real_only:
            before = len(raw_transcripts)
            raw_transcripts = [
                t for t in raw_transcripts
                if t.get('source', '').upper() != 'SYNTHETIC'
            ]
            synthetic_count = before - len(raw_transcripts)
            print(f"🔍 Real-only mode: kept {len(raw_transcripts)} real, "
                f"skipped {synthetic_count} synthetic transcripts")
            print()
        
        if not raw_transcripts:
            print("⚠️  No transcripts found after filtering.")
            return []
        
        # Clean each transcript
        cleaned_transcripts = []
        
        print("🧹 Cleaning transcripts...")
        for transcript in tqdm(raw_transcripts, desc="Cleaning"):
            cleaned = self.clean_transcript(transcript)
            
            if cleaned:
                cleaned_transcripts.append(cleaned)
                self.cleaning_stats['successfully_cleaned'] += 1
            else:
                self.cleaning_stats['failed'] += 1
            
            self.cleaning_stats['total_processed'] += 1
        
        print()
        
        # Calculate averages
        if cleaned_transcripts:
            total_original = sum(t['full_text_original_length'] for t in cleaned_transcripts)
            total_cleaned = sum(t['full_text_cleaned_length'] for t in cleaned_transcripts)
            
            self.cleaning_stats['avg_original_length'] = total_original / len(cleaned_transcripts)
            self.cleaning_stats['avg_cleaned_length'] = total_cleaned / len(cleaned_transcripts)
        
        # Display summary
        self._display_summary()
        
        return cleaned_transcripts
    
    def save_cleaned_transcripts(self, cleaned_transcripts):
        """
        Save cleaned transcripts to processed data directory
        """
        if not cleaned_transcripts:
            print("❌ No cleaned transcripts to save")
            return False
        
        try:
            print()
            print("="*60)
            print("💾 SAVING CLEANED TRANSCRIPTS")
            print("="*60)
            print()
            
            # Save as JSON (full data)
            json_path = self.output_dir / 'transcripts_cleaned.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_transcripts, f, indent=2, ensure_ascii=False)
            
            print(f"✅ JSON saved: {json_path}")
            print(f"   Size: {json_path.stat().st_size / (1024*1024):.2f} MB")
            
            # Save metadata as CSV (for easy viewing)
            metadata_fields = [
                'ticker', 'company_name', 'quarter', 'date',
                'full_text_original_length', 'full_text_cleaned_length',
                'total_chars_removed', 'cleaning_timestamp'
            ]
            
            df = pd.DataFrame(cleaned_transcripts)[metadata_fields]
            csv_path = self.output_dir / 'transcripts_cleaned_metadata.csv'
            df.to_csv(csv_path, index=False)
            
            print(f"✅ Metadata CSV saved: {csv_path}")
            
            # Save cleaning statistics — written AFTER the loop so counts are final
            stats_path = self.output_dir / 'cleaning_statistics.json'
            with open(stats_path, 'w') as f:
                json.dump(self.cleaning_stats, f, indent=2)

            # Also write to logs dir so downstream phases (e.g. Phase 2E) always
            # find the populated version regardless of which path they read.
            log_stats_path = self.log_file.parent / 'cleaning_statistics.json'
            with open(log_stats_path, 'w') as f:
                json.dump(self.cleaning_stats, f, indent=2)
            
            print(f"✅ Statistics saved: {stats_path}")
            print()
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving cleaned transcripts: {e}")
            return False
    
    def _display_summary(self):
        """Display cleaning summary"""
        print("="*60)
        print("📊 CLEANING SUMMARY")
        print("="*60)
        print()
        print(f"Total processed:        {self.cleaning_stats['total_processed']}")
        print(f"Successfully cleaned:   {self.cleaning_stats['successfully_cleaned']}")
        print(f"Failed:                 {self.cleaning_stats['failed']}")
        print()
        print(f"Average original length: {self.cleaning_stats['avg_original_length']:.0f} chars")
        print(f"Average cleaned length:  {self.cleaning_stats['avg_cleaned_length']:.0f} chars")
        print(f"Total chars removed:     {self.cleaning_stats['total_chars_removed']:,}")
        print(f"Reduction:               {(1 - self.cleaning_stats['avg_cleaned_length'] / self.cleaning_stats['avg_original_length']) * 100:.1f}%")
        print()
    
    def _log_error(self, message):
        """Log errors"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()} - ERROR: {message}\n")
    
    def run(self, real_only=False):  # ← Add parameter
        """
        Execute complete transcript cleaning workflow
        
        Args:
            real_only: If True, only process real transcripts
        """
        # Clean transcripts
        cleaned_transcripts = self.clean_all_transcripts(real_only=real_only)
        
        if cleaned_transcripts is None:
            print("❌ Cleaning failed")
            return False
        
        # Save cleaned transcripts
        success = self.save_cleaned_transcripts(cleaned_transcripts)
        
        if success:
            print("="*60)
            print("✅ PHASE 2A COMPLETE")
            print("="*60)
            print()
            print(f"📁 Cleaned data saved to: {self.output_dir}")
            print(f"📝 Log saved to: {self.log_file}")
            print()
            print("✅ Ready for Phase 2B: Speaker Segmentation")
            print()
            
        return success

def main():
    """Main execution"""
    cleaner = TranscriptCleaner()
    success = cleaner.run()
    return success

if __name__ == "__main__":
    main()