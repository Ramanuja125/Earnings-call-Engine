"""
Phase 0.3: Verify all installations
"""

import sys
import importlib
from pathlib import Path

def verify_installation():
    """
    Verifies that all required packages are installed correctly
    """
    
    # List of required packages
    required_packages = {
        'numpy': '1.26.0',
        'pandas': '2.1.0',
        'requests': '2.31.0',
        'bs4': '4.12.0',  # beautifulsoup4
        'selenium': '4.15.0',
        'yfinance': '0.2.0',
        'transformers': '4.35.0',
        'torch': '2.1.0',
        'sklearn': '1.3.0',  # scikit-learn
        'xgboost': '2.0.0',
        'shap': '0.44.0',
        'matplotlib': '3.8.0',
        'seaborn': '0.13.0',
        'tqdm': '4.66.0',
        'streamlit': '1.30.0',
    }
    
    print("="*60)
    print("VERIFYING INSTALLATION")
    print("="*60)
    print()
    
    # Check Python version first
    print(f"🐍 Python version: {sys.version}")
    py_version = sys.version_info
    
    if py_version.major == 3 and py_version.minor >= 12:
        print(f"✅ Python {py_version.major}.{py_version.minor} detected (compatible)")
    elif py_version.major == 3 and py_version.minor >= 8:
        print(f"⚠️  Python {py_version.major}.{py_version.minor} detected (should work)")
    else:
        print(f"❌ Python {py_version.major}.{py_version.minor} detected (not recommended)")
        print("   Recommended: Python 3.8+")
    
    print()
    
    failed = []
    passed = []
    
    for package, min_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            
            # Special handling for sklearn
            if package == 'sklearn':
                import sklearn
                version = sklearn.__version__
            
            print(f"✅ {package:20s} version: {version}")
            passed.append(package)
            
        except ImportError as e:
            print(f"❌ {package:20s} FAILED: {e}")
            failed.append(package)
    
    print("\n" + "="*60)
    print(f"SUMMARY: {len(passed)}/{len(required_packages)} packages verified")
    print("="*60)
    
    if failed:
        print(f"\n❌ Failed packages: {', '.join(failed)}")
        print("Please reinstall these packages manually.")
        return False
    else:
        print("\n✅ All packages verified successfully!")
        
        # Check project structure
        print("\n📁 Checking project structure...")
        required_dirs = ['data', 'src', 'notebooks', 'models', 'results', 'logs']
        all_exist = True
        for dir_name in required_dirs:
            if Path(dir_name).exists():
                print(f"✅ {dir_name}/ exists")
            else:
                print(f"❌ {dir_name}/ missing")
                all_exist = False
        
        if all_exist:
            print("\n" + "="*60)
            print("🎉 ENVIRONMENT SETUP COMPLETE!")
            print("="*60)
            print("\nYou are ready to begin Phase 1: Data Collection")
            print("Waiting for your confirmation to proceed...")
        else:
            print("\n❌ Some directories are missing. Run setup_project.py first.")
        
        return True

if __name__ == "__main__":
    verify_installation()
