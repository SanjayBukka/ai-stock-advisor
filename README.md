# AI Stock Advisor

## Overview
AI Stock Advisor is a machine learning-based tool that predicts stock trends and provides investment suggestions based on historical data. It uses deep learning models to analyze stock price movements and generate insights for traders and investors.

## Features
- Fetches real-time stock data using `yfinance`
- Preprocesses and visualizes historical stock trends
- Implements LSTM (Long Short-Term Memory) neural networks for stock price prediction
- Provides interactive UI using `Streamlit`
- Allows users to input stock symbols and view predictions

## Technologies Used
- Python
- Streamlit
- TensorFlow/Keras
- Scikit-learn
- Matplotlib & Seaborn
- Pandas & NumPy
- yFinance API

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-stock-advisor.git
   ```
2. Navigate to the project folder:
   ```bash
   cd ai-stock-advisor
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the Streamlit web app.
2. Enter the stock ticker symbol (e.g., AAPL, TSLA, GOOG).
3. View stock trend graphs and predicted prices.

## Folder Structure
```
AI-Stock-Advisor/
├── app.py                 # Main Streamlit app
├── model.py               # LSTM model implementation
├── data/                  # Historical stock data
├── requirements.txt       # Required dependencies
├── README.md              # Project documentation
└── .gitignore             # Git ignore file
```

## Future Enhancements
- Improve model accuracy with additional indicators (MACD, RSI, etc.)
- Add more machine learning models for better predictions
- Enhance UI/UX with more interactive visualizations

## Contributing
Feel free to contribute to this project by submitting issues or pull requests.

## License
This project is licensed under the MIT License.

