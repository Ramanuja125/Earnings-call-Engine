# Earnings Call and Risk Intelligence Engine

**FSE 570 Capstone Project**  
Arizona State University

## 📋 Project Overview

An AI-powered system that analyzes earnings call transcripts and financial data to predict post-earnings stock performance. The system combines traditional NLP (FinBERT), financial metrics, and agentic AI to generate actionable investment insights.

## 👥 Team Members

- Shalin N. Bhavsar - sbhavsa8@asu.edu
- Ramanuja M. A. Krishna - rkrish79@asu.edu
- Freya H. Shah - fshah14@asu.edu
- Kruthika Suresh - ksures21@asu.edu
- Harihara Y. Veldanda - hveldan1@asu.edu

## 🎯 Problem Statement

Every year, 10,000+ earnings calls generate massive unstructured text data. Manual analysis is time-consuming and doesn't scale. This project develops an automated system to:
- Extract sentiment and risk signals from earnings transcripts
- Integrate unstructured text with structured financial data
- Predict abnormal stock returns post-earnings
- Provide interpretable insights for investment decisions

## 🏗️ Architecture
Data Collection → Preprocessing → Feature Engineering → ML Modeling → Prediction
↓                ↓                  ↓                  ↓              ↓

Market Data    - Cleaning        - FinBERT         - XGBoost      ROC-AUC
Financials     - Alignment       - LM Dict         - Random Forest Improvement
Transcripts    - Validation      - Agent AI        - Ensemble     ≥3%


## 🚀 Key Features

- **Data Collection**: 503 S&P 500 companies, 738K price records, 3.1K quarterly financials, 150 transcripts
- **NLP Pipeline**: FinBERT embeddings (768-dim) + sentiment analysis
- **Agentic AI**: Multi-agent system for advanced feature extraction (Week 9-10)
- **ML Models**: Logistic Regression baseline, XGBoost primary, Random Forest validation
- **Explainability**: SHAP values for feature importance

## 📊 Dataset

| Data Source | Records | Time Period | Status |
|-------------|---------|-------------|--------|
| Stock Prices | 738,698 | 2018-2023 | ✅ Complete |
| Financial Statements | 3,131 | 2018-2023 | ✅ Complete |
| Earnings Transcripts | 150 | 2022-2023 | ✅ Complete |
| S&P 500 Index | 1,509 | 2018-2023 | ✅ Complete |

## 🛠️ Technology Stack

**Languages & Frameworks:**
- Python 3.12
- PyTorch 2.7
- Transformers (Hugging Face)
- scikit-learn 1.8

**Key Libraries:**
- **NLP**: FinBERT, NLTK, spaCy
- **ML**: XGBoost, LightGBM, scikit-learn
- **Data**: pandas, numpy, yfinance
- **Visualization**: matplotlib, seaborn, plotly
- **Agents**: Anthropic Claude API (Phase 2)

```text
## 📁 Project Structure
.
├── data/
│   ├── raw/              # Original collected data
│   ├── processed/        # Cleaned and normalized data
│   ├── features/         # Extracted features
│   └── final/            # Model-ready datasets
├── src/
│   ├── data_collection/  # Scrapers and collectors
│   ├── preprocessing/    # Data cleaning pipelines
│   ├── features/         # Feature engineering
│   ├── agents/           # Agentic AI modules (Week 9)
│   ├── models/           # ML model implementations
│   └── evaluation/       # Metrics and validation
├── notebooks/            # Jupyter notebooks for EDA
├── models/               # Saved model files
├── results/              # Figures, tables, reports
├── logs/                 # Execution logs
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
├── config.py             # Configuration settings
└── README.md
```


## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/earnings-intelligence-engine.git
cd earnings-intelligence-engine
```

2. Running the entire Pipeline
## Running on Windows
```bash
run_everything.bat
```
## Running on MacOS
```bash
sh run_everything.sh
```

#OR Running manually
2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Verify installation
```bash
python verify_installation.py
```

## 🎯 Usage

### Phase 1: Data Collection
```bash
# Collect all data sources
python run_phase1ab.py  # Market data
python run_phase1c.py   # Financial statements
python run_phase1d.py   # Earnings transcripts
```

### Phase 2: Preprocessing (Coming Soon)
```bash
python run_phase2.py
```

### Phase 3: Feature Engineering (Coming Soon)
```bash
python run_phase3.py
```

### Phase 4-5: Baseline Modeling & Regression & Classification
```bash
python run_phase4.py
```
### Phase 6: Baseline Models Comparison
```bash
python run_phase6.py
```
### Phase 7: XGBoost Models
```bash
python run_phase7.py
```
### Phase 8: Model Comparison & Visualization using Streamlit for UI
```bash
python run_phase8.py
```
### Phase 9: Agentic AI Feature Engineering
```bash
python run_phase9.py
```
### Phase 10: Agent-Enhanced Modeling
```bash
python run_phase10.py
```
### Phase 11: Agentic AI Orchestration Layer
```bash
python run_phase11.py
```

## 📈 Expected Results

| Model | Features | Target ROC-AUC | Status |
|-------|----------|----------------|--------|
| Baseline (Financial Only) | 28 | ≥0.55 | Week 5 |
| Enhanced (FinBERT) | 796 | ≥0.58 | Week 7 |
| Agent-Enhanced | 846 | ≥0.61 | Week 10 |

## 🧪 Validation Strategy

- **Train/Test Split**: 70/30 chronological
- **Cross-Validation**: 5-fold time-series CV
- **Metrics**: Accuracy, F1, ROC-AUC
- **Statistical Test**: DeLong test (p < 0.05)
- **Baseline Comparison**: Sentiment-only vs. Financial-only

## 📊 Current Progress

- [x] Phase 0: Environment Setup (Week 1)
- [x] Phase 1A: S&P 500 Tickers (Week 2)
- [x] Phase 1B: Market Data Collection (Week 2)
- [x] Phase 1C: Financial Statements (Week 3)
- [x] Phase 1D: Transcript Collection (Week 3)
- [x] Phase 2: Data Preprocessing (Week 4-5)
- [x] Phase 3: Feature Engineering (Week 6-7)
- [x] Phase 4-8: ML Modeling and adding User Interface - Streamlit(Week 8-9)
- [x] Phase 9-10: Agentic Enhancement (Week 10-11)

## 🤖 Agentic AI Enhancement (Phase 2)

**Multi-Agent System:**
- **Transcript Analyzer**: Extracts key insights, forward-looking statements
- **Risk Assessor**: Identifies red flags and risk mentions
- **Financial Context Agent**: Analyzes metrics in industry context
- **Orchestrator**: Synthesizes agent findings

**Agent-Derived Features:** Management confidence, competitive pressure, strategic shifts, risk severity

## 📚 References

- Bernard, V. L., & Thomas, J. K. (1989). Post-earnings-announcement drift
- FinBERT: Financial Sentiment Analysis (Araci, 2019)
- Loughran-McDonald Financial Dictionary (2011)

## 📝 License

This project is part of academic coursework at Arizona State University.

## 🙏 Acknowledgments

- Arizona State University, FSE 570
- Anthropic (Claude API for agentic AI)
- Hugging Face (FinBERT model)

## 📧 Contact

For questions or collaboration:
sbhavsa8@asu.edu - Shalin N Bhavsar
rkrish79@asu.edu - Ramanuja M.A Krishna
fshah14@asu.edu - Freya Shah
ksures21@asu.edu - Kruthika Suresh
hveldan1@asu.edu - Harihara Y. Veldanda
---

**Status**: 🟢 Complete
