# Stock Market Forecasting

📈 Stock Market Forecasting
An interactive Streamlit dashboard for analyzing and forecasting stock prices using classical time series models (ARIMA, SARIMA, Prophet) and deep learning (LSTM). This tool supports real-time data fetching from Yahoo Finance or CSV uploads and provides intuitive visualizations with accuracy metrics for effective decision-making.

📁 Features
✅ Upload your own CSV stock dataset
✅ Fetch historical stock data using ticker symbols (e.g., AAPL, TSLA, INFY.NS)
📈 View interactive line charts of historical closing prices
🔮 Forecast using multiple models:
-ARIMA
-SARIMA
-Facebook Prophet
-LSTM (Long Short-Term Memory)

🧮 View forecast accuracy with metrics:
-Mean Absolute Error (MAE)
-Root Mean Squared Error (RMSE)

📊 Model comparison & performance interpretation
📄 Raw data preview and cleanup
📥 Optional: Upload your own dataset in CSV format

📦 Tech Stack
-Python 3.x
-Streamlit
-Facebook Prophet
-TensorFlow/Keras (for LSTM)
-scikit-learn
-statsmodels
-Pandas, NumPy
-Matplotlib, Seaborn, Plotly
-yfinance (for fetching stock data)

🚀 How to Run
~ Clone the repository:
git clone https://github.com/your-username/stock-forecast-dashboard.git  
cd stock-forecast-dashboard

~ Create virtual environment and activate:
python -m venv venv  
venv\Scripts\activate     # On Windows

~ Install required packages:
pip install the following in your terminal
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
statsmodels
prophet
tensorflow
keras
streamlit

~ Run the Streamlit app:
streamlit run App.py

📂 Dataset Format
Your dataset should be in CSV format and must include:
-Date (or ds) — the date of the record
-Close (or y) — the closing price of the stock
Or, enter a stock ticker (like AAPL, TSLA, INFY.NS) to fetch real-time data using Yahoo Finance.

Example:
Date,Close  
2023-01-01,125.34  
2023-01-02,127.89  
...

🧪 Output Samples

-Forecasted price chart (actual vs predicted)
-Model performance (MAE, RMSE)
-Interactive forecast horizon selector
-Expandable raw data table
