# Stock Market Predictor

## Streamlit FrontEnd UI

![screencapture-localhost-8501-2024-11-26-10_30_53](https://github.com/user-attachments/assets/951915a6-da19-4896-8827-72c2cd01904f)

## Overview

The Stock Market Predictor is an advanced tool designed to predict stock prices using Long Short-Term Memory (LSTM) models and provide an intuitive user interface for real-time stock market analysis. The application integrates historical stock data, moving averages, Fibonacci retracement levels, and real-time news updates to offer a comprehensive solution for investors and traders.

## Features

- **Historical Stock Data Retrieval**: Fetches historical stock price data from Yahoo Finance.
- **Data Preprocessing**: Scales and preprocesses the data for LSTM model training.
- **LSTM Model Training**: Utilizes LSTM neural networks to predict future stock prices.
- **Interactive User Interface**: Built using Streamlit, allowing users to input stock symbols and visualize data.
- **Moving Averages**: Displays moving averages (50-day, 100-day, and 200-day) for trend analysis.
- **Fibonacci Retracement Levels**: Calculates and visualizes Fibonacci retracement levels.
- **Real-Time News Updates**: Integrates real-time news related to the selected stock.

## Installation

To set up the Stock Market Predictor, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock-market-predictor.git
   cd stock-market-predictor
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Input Stock Symbol**: Enter the stock symbol (e.g., GOOG for Google) in the input field.
2. **View Stock Information**: The app fetches and displays essential stock information and visualizes historical data.
3. **Analyze Moving Averages**: Select from various moving average graphs to analyze stock trends.
4. **Check Fibonacci Retracement Levels**: View calculated Fibonacci levels for potential support and resistance areas.
5. **Read Recent News**: Access real-time news updates related to the selected stock to stay informed about market events.

## Project Structure

- `app.py`: Main application file containing the Streamlit app code.
- `models/`: Directory containing the LSTM model and other machine learning scripts.
- `data/`: Folder for storing fetched and preprocessed data.
- `requirements.txt`: List of dependencies required to run the project.
- `README.md`: Project documentation file.

## System Requirements

- **Python 3.7 or higher**
- **pip (Python package installer)**
- **Internet connection** (to fetch real-time data from Yahoo Finance)

## Contributing

We welcome contributions to enhance the Stock Market Predictor. To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push them to your forked repository.
4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

We extend our gratitude to the contributors and maintainers of the following libraries and tools:

- [Yahoo Finance](https://pypi.org/project/yfinance/)
- [Streamlit](https://streamlit.io/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)

## Contact

For any questions or feedback, please contact me at [adityamore896@hotmail.com].

---

By combining machine learning techniques with a user-friendly interface, the Stock Market Predictor aims to empower investors with valuable insights and tools for making informed decisions in the complex world of stock markets.
