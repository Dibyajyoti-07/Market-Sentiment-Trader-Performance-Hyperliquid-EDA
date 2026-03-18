# Trader Intelligence Dashboard: Market Sentiment vs. Performance

### 🔗 [Live Dashboard](https://market-sentiment-trader-performance-hyperliquid-eda-znbpk3dbr7.streamlit.app/)

An Advanced Exploratory Data Analysis (EDA) dashboard that investigates the relationship between market sentiment (Crypto Fear & Greed Index) and historical trader performance on Hyperliquid.

## 📊 Overview
This project analyzes hundreds of thousands of individual trades to determine if market sentiment acts as a reliable predictor or a behavioral trap. It features a modern, interactive Streamlit dashboard with deep-dive analytics into win rates, PnL distribution, and trader archetypes.

## 🚀 Key Features
- **Sentiment Segments:** Analysis across Extreme Fear, Fear, Neutral, Greed, and Extreme Greed.
- **Long vs. Short:** Directional performance tracking across different market regimes.
- **Trader Profiling:** Automatic classification of "Smart Money" vs. "Retail" based on net profitability.
- **Whale Tracking:** Isolation and analysis of the top 5% largest trades by USD size.
- **AI Trade Predictor:** A Deep Neural Network trained on 86,000+ profitable trades to forecast Long vs Short profitability based on sentiment, position size, and time.
- **Interactive Explorer:** Dynamic filters for Coins, Date Ranges, and Trade Sides with CSV export capabilities.

## 🛠️ Tech Stack
- **Language:** Python 3.11+
- **Frontend/Dashboard:** [Streamlit](https://streamlit.io/)
- **Visuals:** [Plotly](https://plotly.com/python/)
- **Data:** Pandas, Numpy, Scikit-Learn
- **Machine Learning:** TensorFlow, Keras
- **Environment:** Jupyter Notebook (for raw data pipeline)

## 🤖 AI Model Details: Trade Predictor NN

The dashboard features a **Deep Neural Network (DNN)** trained on a massive dataset of high-performance trades to provide real-time directional recommendations.

### 🧠 Architecture
The model is a **Feed-Forward Neural Network (FNN)** built using Keras/TensorFlow with the following structure:
- **Input Layer:** 196 features (after One-Hot Encoding categorical coins and scaling numeric inputs).
- **Hidden Layer 1:** 256 units (ReLU) + Batch Normalization + Dropout (30%).
- **Hidden Layer 2:** 128 units (ReLU) + Batch Normalization + Dropout (25%).
- **Hidden Layer 3:** 64 units (ReLU) + Dropout (20%).
- **Hidden Layer 4:** 32 units (ReLU).
- **Output Layer:** 1 unit (Sigmoid) — outputs probability of a "Long" trade being profitable.

### 📊 Training & Performance
- **Dataset:** Trained on **86,863 historically profitable trades** ("Winners") to learn the DNA of a successful entry.
- **Optimization:** Adam Optimizer (LR=0.001) with Binary Cross-Entropy loss.
- **Primary Accuracy:** ~56.5% on completely unseen test data (significantly outperforming a random 50/50 baseline).
- **Baseline Comparison:** Outperforms the Random Forest Classifier (~59% on training but higher variance on test).

### ⚙️ Feature Engineering
The model analyzes 10+ core signals for every prediction:
1. **Market Sentiment:** Raw Fear & Greed Index (0-100).
2. **Sentiment Rank:** Categorical classification (Extreme Fear to Extreme Greed).
3. **Log Size (USD):** Log-normalized trade size to handle whale outliers.
4. **Temporal Flags:** Hour of day, Day of week, and Weekend status.
5. **Coin Metrics:** Individual coin win-rates and size percentiles.
6. **Interaction Effects:** Sentiment × Position Size synergies.

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
- `models/`: Serialized Keras `.h5` model and `.pkl` preprocessor pipeline.
- `README.md`: Project documentation.
- `LICENSE`: MIT License.

## 💡 Findings
- **Smart Money** accounts show higher activity during "Extreme Fear," exploiting capitulation.
- **Retail** traders frequently "average down" with larger sizes during panics, leading to significant realized losses.
- **Extreme Greed** typically yields the highest win rates but also the highest PnL variance, indicating high-risk momentum chasing.

## ⚠️ Deployment Troubleshooting (Streamlit Cloud)
If you encounter a `No matching distribution found for tensorflow` or resolving dependencies error on Streamlit Community Cloud:
1. Streamlit Cloud may default to Python 3.14+ for new deployments, but **TensorFlow requires Python 3.9 - 3.12**.
2. Go to your Streamlit App Dashboard.
3. Click the **three dots (⋮)** next to your app -> **Settings**.
4. Go to the **Advanced settings** tab.
5. Change the **Python version** dropdown from 3.14 to **3.11** or **3.12**.
6. At the bottom, click **Save**, and then reboot your app.

---
*Created for Advanced Trader Analytics.*
