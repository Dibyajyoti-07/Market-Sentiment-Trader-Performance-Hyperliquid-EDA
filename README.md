# Trader Intelligence Dashboard: Market Sentiment vs. Performance

An Advanced Exploratory Data Analysis (EDA) dashboard that investigates the relationship between market sentiment (Crypto Fear & Greed Index) and historical trader performance on Hyperliquid.

## 📊 Overview
This project analyzes hundreds of thousands of individual trades to determine if market sentiment acts as a reliable predictor or a behavioral trap. It features a modern, interactive Streamlit dashboard with deep-dive analytics into win rates, PnL distribution, and trader archetypes.

## 🚀 Key Features
- **Sentiment Segments:** Analysis across Extreme Fear, Fear, Neutral, Greed, and Extreme Greed.
- **Long vs. Short:** Directional performance tracking across different market regimes.
- **Trader Profiling:** Automatic classification of "Smart Money" vs. "Retail" based on net profitability.
- **Whale Tracking:** Isolation and analysis of the top 5% largest trades by USD size.
- **Interactive Explorer:** Dynamic filters for Coins, Date Ranges, and Trade Sides with CSV export capabilities.

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **Frontend/Dashboard:** [Streamlit](https://streamlit.io/)
- **Visuals:** [Plotly](https://plotly.com/python/)
- **Data:** Pandas, Numpy
- **Environment:** Jupyter Notebook (for raw data pipeline)

## 📦 Installation & Setup

1. **Clone the repository** (or navigate to the project folder):
   ```bash
   cd d:/EDA
   ```

2. **Create and Activate Virtual Environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Dashboard**:
   ```bash
   streamlit run app.py
   ```

## 📁 Project Structure
- `app.py`: Main Streamlit dashboard application.
- `eda_analysis.ipynb`: Raw analysis and data merging pipeline.
- `requirements.txt`: List of Python dependencies.
- `Datasets/`: Historical trade data and sentiment index files.
- `README.md`: Project documentation.
- `LICENSE`: MIT License.

## 💡 Findings
- **Smart Money** accounts show higher activity during "Extreme Fear," exploiting capitulation.
- **Retail** traders frequently "average down" with larger sizes during panics, leading to significant realized losses.
- **Extreme Greed** typically yields the highest win rates but also the highest PnL variance, indicating high-risk momentum chasing.

---
*Created for Advanced Trader Analytics.*
