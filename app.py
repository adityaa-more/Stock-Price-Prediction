import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model('C:/Users/adity/OneDrive/Desktop/Stock Price Prediction/Stock Predictions Model.keras')

st.title('Stock Market Predictor')

stock =st.text_input('Enter Stock Symbol', 'GOOG')
start = '2014-01-01'
end = '2024-02-28'

try:
    data = yf.download(stock, start, end)
except Exception as e:
    st.write(f"Error fetching data: {e}")

if data.empty:
    st.write("No data fetched. Symbol may be incorrect or no data available for the specified date range.")
else:

    infor = yf.Ticker(stock).info
    news = yf.Ticker(stock).news

    st.subheader(infor["longName"])
    col1, col2 = st.columns(2)

    with col1:
        st.write('Quote Type :', infor["quoteType"])
        st.write('Exchange :', infor["exchange"])

    with col2:
        if "currentPrice" in infor:
            st.write('Current Price :', infor["currentPrice"], infor["currency"])
        else:
            st.write('Currency :', infor["currency"])
        st.write('TimeZone : ', infor["timeZoneFullName"])

    #------------------------------------------------------------------------------------------------------------

    st.text(" ")
    st.subheader('Stock Data')
    st.write(data)

    #------------------------------------------------------------------------------------------------------------

    st.text(" ")
    st.subheader('Moving Averages')

    def plot_graph(data, ma_50_days=None, ma_100_days=None, ma_200_days=None):
        fig = plt.figure(figsize=(8,6))
        if ma_50_days is not None:
            plt.plot(ma_50_days, 'r', label='MA 50')
        if ma_100_days is not None:
            plt.plot(ma_100_days, 'b', label='MA 100')
        if ma_200_days is not None:
            plt.plot(ma_200_days, 'r', label='MA 200')
        plt.plot(data.Close, 'g', label='Closing Price')
        plt.legend()  
        plt.title('Moving Average and Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True) 
        st.pyplot(fig)

    def calculate_moving_averages(data):
        ma_50_days = data.Close.rolling(50).mean()
        ma_100_days = data.Close.rolling(100).mean()
        ma_200_days = data.Close.rolling(200).mean()
        return ma_50_days, ma_100_days, ma_200_days

    def plot_price_vs_ma50(data):
        ma_50_days, _, _ = calculate_moving_averages(data)
        plot_graph(data, ma_50_days=ma_50_days)

    def plot_ma50_vs_ma100(data):
        ma_50_days, ma_100_days, _ = calculate_moving_averages(data)
        plot_graph(data, ma_50_days=ma_50_days, ma_100_days=ma_100_days)

    def plot_ma100_vs_ma200(data):
        _, ma_100_days, ma_200_days = calculate_moving_averages(data)
        plot_graph(data, ma_100_days=ma_100_days, ma_200_days=ma_200_days)

    selected_graph = st.selectbox('Select Graph', ['Price vs MA50', 'MA50 vs MA100', 'MA100 vs MA200'])

    if selected_graph == 'Price vs MA50':
        plot_price_vs_ma50(data)
    elif selected_graph == 'MA50 vs MA100':
        plot_ma50_vs_ma100(data)
    elif selected_graph == 'MA100 vs MA200':
        plot_ma100_vs_ma200(data)


    #------------------------------------------------------------------------------------------------------------

    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Predicted Price')
    plt.plot(y, 'g', label = 'Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()
    st.pyplot(fig4)

    #------------------------------------------------------------------------------------------------------------

    st.text(" ")
    st.subheader('Fibonacci Retracement Levels')
    
    latest_2_years_data = data.tail(2 * 365) 

    highest_value = latest_2_years_data['Close'].max()
    lowest_value = latest_2_years_data['Close'].min()

    highest_date = latest_2_years_data.loc[latest_2_years_data['Close'] == highest_value].index[0]
    lowest_date = latest_2_years_data.loc[latest_2_years_data['Close'] == lowest_value].index[0]

    retracement_levels = {
        0: highest_value,
        23.6: highest_value - 0.236 * (highest_value - lowest_value),
        38.2: highest_value - 0.382 * (highest_value - lowest_value),
        50: (highest_value + lowest_value) / 2,
        61.8: highest_value - 0.618 * (highest_value - lowest_value),
        100: lowest_value
    }

    fig = plt.figure(figsize=(10, 6))
    plt.plot(latest_2_years_data.index, latest_2_years_data['Close'], label='Close Price')
    plt.scatter(highest_date, highest_value, color='red', marker='o', label='Highest Value')
    plt.scatter(lowest_date, lowest_value, color='green', marker='o', label='Lowest Value')
    plt.plot([highest_date, lowest_date], [highest_value, lowest_value], color='gray', linestyle='--')

    for level, price in retracement_levels.items():
        plt.axhline(price, color='orange', linestyle='--', label=f'{level}% Fibonacci')

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Fibonacci Retracement Levels')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(fig)

    #------------------------------------------------------------------------------------------------------------

    col1, col2 = st.columns([1, 3]) 
    with col1:
        st.subheader('Recent news')
    with col2:
        show_news = st.button(label=":arrow_double_down:")

    ##### make this in streamlit => store image and other data in container
    
    def create_card(title, publisher, thumbnail_url, link):
        card_html = f"""
        <div style="background-color: #F0F0F0; font-size: 0.875em; border-radius: 15px; padding: 10px; margin-top: 20px;box-shadow: rgba(149, 157, 165, 0.2) 0px 8px 24px;">
            <div style="display: flex;">
                <div style="flex: 30%; height: 125px; margin: 10px; border-radius: 15px; overflow: hidden;">
                    <img src="{thumbnail_url}" style="width: 100%; height: 100%; object-fit: fill;">
                </div>
                <div style="flex: 70%; padding: 10px; color: black;">
                    <h5>{title}</h5>
                    <p>{publisher}</p>
                    <a href="{link}" target="_blank" style="color: #009FFF;">Read more..</a>
                </div>
            </div>
        </div>
        """
        return card_html

    # Determine whether to display news cards based on button state
    if show_news:
        for i, article in enumerate(news[:5]):
            uuid = article['uuid']
            title = article['title']
            publisher = article['publisher']
            link = article['link']
            
            # Check if the article has a thumbnail
            if 'thumbnail' in article and 'resolutions' in article['thumbnail'] and article['thumbnail']['resolutions']:
                thumbnail_url = article['thumbnail']['resolutions'][0]['url']
            else:
                thumbnail_url = None  # Set thumbnail_url to None if thumbnail is not found
            
            
            # Create the card HTML
            card_html = create_card(title, publisher, thumbnail_url, link)
            
            # Display the card
            st.markdown(card_html, unsafe_allow_html=True)