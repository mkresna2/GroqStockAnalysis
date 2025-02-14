import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import groq
import tempfile
import base64
import os
from datetime import datetime, timedelta
import ta  # Technical Analysis Library (for RSI, MACD, Stochastic, ATR, etc.)

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI Stock Analysis with LLaMA 3.2 Vision")
st.sidebar.header("Configuration")

# Input for stock ticker
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., NVDA):", "NVDA")

# Date range selection
st.sidebar.subheader("Date Range")
date_range_option = st.sidebar.selectbox(
    "Select Date Range:",
    ["Custom", "Last 30 Days", "Last 90 Days", "Last 180 Days", "Last 365 Days"],
    index=0
)

# Set start_date and end_date based on the selected option
end_date = datetime.today()
if date_range_option == "Custom":
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", value=end_date)
else:
    days = int(date_range_option.split()[1])  # Extract the number of days from the option
    start_date = end_date - timedelta(days=days)
    st.sidebar.write(f"Automatically set start date to: {start_date.strftime('%Y-%m-%d')}")

# Cache stock data to avoid redundant downloads
@st.cache_data
def fetch_stock_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the given ticker and date range.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Fetch stock data
if st.sidebar.button("Fetch Data"):
    st.session_state["stock_data"] = fetch_stock_data(ticker, start_date, end_date)
    if st.session_state["stock_data"] is not None:
        st.success("Stock data loaded successfully!")

# Check if data is available
if "stock_data" in st.session_state and st.session_state["stock_data"] is not None:
    data = st.session_state["stock_data"]

    # Debug: Display the first few rows of the data
    #st.write("Data loaded successfully!")
    #st.write(data.head())

    # Debug: Display column names
    #st.write("Column names in the data:")
    #st.write(data.columns.tolist())

    # Extract the actual ticker symbol from the column names
    # The second level of the MultiIndex contains the ticker symbol
    actual_ticker = data.columns[0][1]  # e.g., ('Open', 'ABEV.SA') -> 'ABEV.SA'
    st.write(f"Actual ticker symbol in data: {actual_ticker}")

    # Check if required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if (col, actual_ticker) not in data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
    else:
        # Plot candlestick chart
                
        # Debug: Check if the chart is created
        #st.write("Chart created successfully!")

        # Sidebar: Select technical indicators and customize periods
        st.sidebar.subheader("Technical Indicators")
        indicators = st.sidebar.multiselect(
            "Select Indicators:",
            [
                "SMA", "EMA", "Bollinger Bands", "VWAP", 
                "RSI", "MACD", "Stochastic Oscillator", 
                "ATR", "Ichimoku Cloud", "Fibonacci Retracement", 
                "Parabolic SAR", "OBV"
            ],
            default=["SMA"]
        )

        # Input fields for indicator periods
        indicator_periods = {}
        if "SMA" in indicators:
            indicator_periods["SMA"] = st.sidebar.number_input("SMA Period", min_value=1, value=20)
        if "EMA" in indicators:
            indicator_periods["EMA"] = st.sidebar.number_input("EMA Period", min_value=1, value=20)
        if "Bollinger Bands" in indicators:
            indicator_periods["Bollinger Bands"] = st.sidebar.number_input("Bollinger Bands Period", min_value=1, value=20)
        if "RSI" in indicators:
            indicator_periods["RSI"] = st.sidebar.number_input("RSI Period", min_value=1, value=14)
        if "MACD" in indicators:
            indicator_periods["MACD Fast"] = st.sidebar.number_input("MACD Fast Period", min_value=1, value=12)
            indicator_periods["MACD Slow"] = st.sidebar.number_input("MACD Slow Period", min_value=1, value=26)
            indicator_periods["MACD Signal"] = st.sidebar.number_input("MACD Signal Period", min_value=1, value=9)
        if "Stochastic Oscillator" in indicators:
            indicator_periods["Stochastic %K"] = st.sidebar.number_input("Stochastic %K Period", min_value=1, value=14)
            indicator_periods["Stochastic %D"] = st.sidebar.number_input("Stochastic %D Period", min_value=1, value=3)
        if "ATR" in indicators:
            indicator_periods["ATR"] = st.sidebar.number_input("ATR Period", min_value=1, value=14)
        if "Ichimoku Cloud" in indicators:
            indicator_periods["Ichimoku Conversion"] = st.sidebar.number_input("Ichimoku Conversion Period", min_value=1, value=9)
            indicator_periods["Ichimoku Base"] = st.sidebar.number_input("Ichimoku Base Period", min_value=1, value=26)
            indicator_periods["Ichimoku Lagging"] = st.sidebar.number_input("Ichimoku Lagging Period", min_value=1, value=52)
        if "Fibonacci Retracement" in indicators:
            indicator_periods["Fibonacci Levels"] = st.sidebar.text_input("Fibonacci Levels (comma-separated)", value="0.236, 0.382, 0.5, 0.618, 0.786")
        if "Parabolic SAR" in indicators:
            indicator_periods["Parabolic SAR Step"] = st.sidebar.number_input("Parabolic SAR Step", min_value=0.01, value=0.02, step=0.01)
            indicator_periods["Parabolic SAR Max"] = st.sidebar.number_input("Parabolic SAR Max", min_value=0.01, value=0.2, step=0.01)
        if "OBV" in indicators:
            indicator_periods["OBV"] = None  # OBV does not require a period

        
        # Determine needed subplots based on selected indicators
        subplot_config = {
            'price': 1,
            'stochastic': 2 if "Stochastic Oscillator" in indicators else 0,
            'rsi': 3 if "RSI" in indicators else 0,
            'macd': 4 if "MACD" in indicators else 0,
            'volume': 5 if "OBV" in indicators or "VWAP" in indicators else 0,
            'atr': 6 if "ATR" in indicators else 0,
            'ichimoku': 7 if "Ichimoku Cloud" in indicators else 0  # Add Ichimoku Cloud
        }

        rows = max(subplot_config.values()) if max(subplot_config.values()) > 0 else 1
        row_heights = [0.4] + [0.12]*(rows-1)  # 40% for main chart, 12% for each subplot

        # Create subplots only if data is valid
        fig = make_subplots(
            rows=rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            specs=[[{"secondary_y": True}]]*rows
        )

        # Add base candlestick to main chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data[('Open', actual_ticker)].tolist(),
                high=data[('High', actual_ticker)].tolist(),
                low=data[('Low', actual_ticker)].tolist(),
                close=data[('Close', actual_ticker)].tolist(),
                name="Price"
            ), row=1, col=1
        )

        # Helper function to add indicators to the chart
        def add_indicator(indicator, period=None, row=1):
            try:
                if indicator == "SMA":
                    sma = data[('Close', actual_ticker)].rolling(window=period).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=sma.tolist(), 
                            mode='lines', name=f'SMA ({period})'), row=row, col=1)
                
                elif indicator == "EMA":
                    ema = data[('Close', actual_ticker)].ewm(span=period).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=ema.tolist(), 
                            mode='lines', name=f'EMA ({period})'), row=row, col=1)
                
                elif indicator == "Bollinger Bands":
                    sma = data[('Close', actual_ticker)].rolling(window=period).mean()
                    std = data[('Close', actual_ticker)].rolling(window=period).std()
                    bb_upper = sma + 2 * std
                    bb_lower = sma - 2 * std
                    fig.add_trace(go.Scatter(x=data.index, y=bb_upper.tolist(), 
                            mode='lines', name=f'BB Upper'), row=row, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=bb_lower.tolist(), 
                            mode='lines', name=f'BB Lower'), row=row, col=1)
                
                elif indicator == "VWAP":
                    vwap = (data[('Close', actual_ticker)] * data[('Volume', actual_ticker)]).cumsum() / data[('Volume', actual_ticker)].cumsum()
                    fig.add_trace(go.Scatter(x=data.index, y=vwap.tolist(), 
                            mode='lines', name='VWAP'), row=row, col=1)
                
                elif indicator == "RSI":
                    rsi = ta.momentum.RSIIndicator(data[('Close', actual_ticker)], window=period).rsi()
                    fig.add_trace(go.Scatter(x=data.index, y=rsi.tolist(), mode='lines', 
                            name=f'RSI ({period})', line=dict(color='purple')), 
                            row=subplot_config['rsi'], col=1)
                    fig.update_yaxes(range=[0,100], row=subplot_config['rsi'], col=1)

                elif indicator == "MACD":
                    macd = ta.trend.MACD(
                        data[('Close', actual_ticker)],
                        window_fast=indicator_periods["MACD Fast"],
                        window_slow=indicator_periods["MACD Slow"],
                        window_sign=indicator_periods["MACD Signal"]
                    )
                    fig.add_trace(go.Scatter(x=data.index, y=macd.macd().tolist(), 
                                            mode='lines', name='MACD Line', line=dict(color='blue')), 
                                            row=subplot_config['macd'], col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=macd.macd_signal().tolist(), 
                                            mode='lines', name='Signal Line', line=dict(color='orange')), 
                                            row=subplot_config['macd'], col=1)
                    fig.add_trace(go.Bar(x=data.index, y=macd.macd_diff().tolist(), 
                                        name='MACD Histogram', marker=dict(color='gray')), 
                                        row=subplot_config['macd'], col=1)
                
                elif indicator == "Stochastic Oscillator":
                    stoch = ta.momentum.StochasticOscillator(
                        data[('High', actual_ticker)], data[('Low', actual_ticker)], 
                        data[('Close', actual_ticker)], 
                        window=indicator_periods["Stochastic %K"], 
                        smooth_window=indicator_periods["Stochastic %D"]
                    )
                    fig.add_trace(go.Scatter(x=data.index, y=stoch.stoch().tolist(), 
                            mode='lines', name='Stoch %K', line=dict(color='blue')), 
                            row=subplot_config['stochastic'], col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=stoch.stoch_signal().tolist(), 
                            mode='lines', name='Stoch %D', line=dict(color='orange')), 
                            row=subplot_config['stochastic'], col=1)
                    fig.update_yaxes(range=[0,100], row=subplot_config['stochastic'], col=1)
            
                elif indicator == "ATR":
                    atr = ta.volatility.AverageTrueRange(
                        data[('High', actual_ticker)], data[('Low', actual_ticker)], 
                        data[('Close', actual_ticker)], window=period
                    ).average_true_range()
                    fig.add_trace(go.Scatter(x=data.index, y=atr.tolist(), 
                            mode='lines', name=f'ATR ({period})', 
                            line=dict(color='green')), row=subplot_config['atr'], col=1)

                elif indicator == "Ichimoku Cloud":
                    ichimoku = ta.trend.IchimokuIndicator(
                        data[('High', actual_ticker)],
                        data[('Low', actual_ticker)],
                        window_conversion=indicator_periods["Ichimoku Conversion"],
                        window_base=indicator_periods["Ichimoku Base"],
                        window_lagging=indicator_periods["Ichimoku Lagging"]
                    )
                    # Add Conversion Line (Tenkan-sen)
                    fig.add_trace(go.Scatter(x=data.index, y=ichimoku.ichimoku_conversion_line().tolist(), 
                                            mode='lines', name='Conversion Line', line=dict(color='blue')), 
                                            row=1, col=1)
                    # Add Base Line (Kijun-sen)
                    fig.add_trace(go.Scatter(x=data.index, y=ichimoku.ichimoku_base_line().tolist(), 
                                            mode='lines', name='Base Line', line=dict(color='red')), 
                                            row=1, col=1)
                    # Add Leading Span A (Senkou Span A)
                    fig.add_trace(go.Scatter(x=data.index, y=ichimoku.ichimoku_a().tolist(), 
                                            mode='lines', name='Leading Span A', line=dict(color='green')), 
                                            row=1, col=1)
                    # Add Leading Span B (Senkou Span B)
                    fig.add_trace(go.Scatter(x=data.index, y=ichimoku.ichimoku_b().tolist(), 
                                            mode='lines', name='Leading Span B', line=dict(color='orange')), 
                                            row=1, col=1)
                    # Add Lagging Span (Chikou Span)
                    fig.add_trace(go.Scatter(x=data.index, y=ichimoku.ichimoku_chikou().tolist(), 
                                            mode='lines', name='Lagging Span', line=dict(color='purple')), 
                                            row=1, col=1)
                
                elif indicator == "Parabolic SAR":
                    psar = ta.trend.PSARIndicator(
                        data[('High', actual_ticker)], data[('Low', actual_ticker)], 
                        data[('Close', actual_ticker)], 
                        step=indicator_periods["Parabolic SAR Step"], 
                        max_step=indicator_periods["Parabolic SAR Max"]
                    ).psar()
                    fig.add_trace(go.Scatter(x=data.index, y=psar.tolist(), 
                            mode='markers', name='Parabolic SAR', 
                            marker=dict(color='red', size=4)), row=row, col=1)
                
                elif indicator == "OBV":
                    obv = ta.volume.OnBalanceVolumeIndicator(
                        data[('Close', actual_ticker)], 
                        data[('Volume', actual_ticker)]
                    ).on_balance_volume()
                    fig.add_trace(go.Scatter(x=data.index, y=obv.tolist(), 
                            mode='lines', name='OBV', line=dict(color='blue')), 
                            row=subplot_config['volume'], col=1)
                
                elif indicator == "Fibonacci Retracement":
                    max_price = data[('High', actual_ticker)].max()
                    min_price = data[('Low', actual_ticker)].min()
                    levels = [float(l.strip()) for l in indicator_periods["Fibonacci Levels"].split(",")]
                    for level in levels:
                        fib_level = max_price - (max_price - min_price) * level
                        fig.add_trace(go.Scatter(x=data.index, y=[fib_level]*len(data), 
                                mode='lines', name=f'Fib {level*100}%', 
                                line=dict(dash='dot')), row=row, col=1)

            except Exception as e:
                st.error(f"Error adding {indicator}: {str(e)}")

        # Add selected indicators to the chart
        for indicator in indicators:
            if indicator == "MACD":
                add_indicator(indicator, indicator_periods.get(indicator), row=subplot_config['macd'])
            elif indicator == "Ichimoku Cloud":
                add_indicator(indicator, indicator_periods.get(indicator), row=1)  # Ichimoku is plotted on the main chart
            else:
                add_indicator(indicator, indicator_periods.get(indicator), row=1)

        # Final layout adjustments
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            title=f"{ticker} Analysis - {', '.join(indicators)}",
            height=600 + (100*(rows-1)),
            showlegend=True,
            hovermode='x unified'
        )

        # Set axis titles
        fig.update_yaxes(title_text="Price", row=1, col=1)
        
        # Filter indicators to only include those with subplots
        subplot_indicators = []
        if "Stochastic Oscillator" in indicators:
            subplot_indicators.append("Stochastic Oscillator")
        if "RSI" in indicators:
            subplot_indicators.append("RSI")
        if "MACD" in indicators:
            subplot_indicators.append("MACD")
        if "OBV" in indicators:
            subplot_indicators.append("OBV")
        if "ATR" in indicators:
            subplot_indicators.append("ATR")
        if "Ichimoku Cloud" in indicators:
            subplot_indicators.append("Ichimoku Cloud")

        # Update y-axis titles for subplots
        for r in range(2, rows+1):
            if r-2 < len(subplot_indicators):
                fig.update_yaxes(title_text=subplot_indicators[r-2], row=r, col=1)

        # Render the chart
        st.plotly_chart(fig, use_container_width=True)
        # --- End of chart creation --- #                             

        # Debug: Display the chart configuration
        #st.write("Chart configuration:")
        #st.write(fig.to_dict())

        # Render the chart
        #st.plotly_chart(fig, use_container_width=True)

        # Analyze chart with LLaMA 3.2 Vision
        st.subheader("AI-Powered Analysis")
        if st.button("Run AI Analysis"):
            # Before the tempfile section, add:
            output_path = "E:/Project/StockAnalysis/recordings/"
            os.makedirs(output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = f"{output_path}chart_recording_{timestamp}.html"

            # Save interactive HTML with recording capability
            fig.write_html(
                html_path,
                config={
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'toImageButtonOptions': {'format': 'png', 'filename': 'custom_image'},
                    'modeBarButtonsToAdd': ['drawopenpath', 'eraseshape']
                }
            )
            tmpfile_path = None
            with st.spinner("Analyzing the chart, please wait..."):
                try:
                    # Check for Groq API key
                    api_key = st.secrets.get("GROQ_API_KEY", None)
                    if api_key:
                        # Initialize Groq client
                        from groq import Groq
                        client = Groq(api_key=api_key)
                        
                        # Save chart as a temporary image
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                            fig.write_image(tmpfile.name)
                            tmpfile_path = tmpfile.name

                        # Read image and encode to Base64
                        with open(tmpfile_path, "rb") as image_file:
                            image_data = base64.b64encode(image_file.read()).decode('utf-8')
                            image_url = f"data:image/png;base64,{image_data}"

                        # Create completion request
                        completion = client.chat.completions.create(
                            model="llama-3.2-11b-vision-preview",
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": """You are a Stock Trader specializing in Technical Analysis at a top financial institution.
                                            Analyze the stock chart's technical indicators and provide a buy/hold/sell recommendation.
                                            Base your recommendation only on the candlestick chart and the displayed technical indicators.
                                            First, provide the recommendation, then, provide your detailed reasoning."""
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": image_url
                                            }
                                        }
                                    ]
                                }
                            ],
                            temperature=0.7,
                            max_completion_tokens=1024,
                            top_p=1,
                            stream=False,
                            stop=None
                        )

                        # Display AI analysis result
                        st.write("**AI Analysis Results:**")
                        st.write(completion.choices[0].message.content)
                    else:
                        st.error("Please set up your Groq API key in Streamlit secrets. Visit https://console.groq.com to get your API key.")
                except Exception as e:
                    st.error(f"Error during AI analysis: {e}")
                finally:
                    # Clean up temporary file
                    if tmpfile_path and os.path.exists(tmpfile_path):
                        try:
                            os.remove(tmpfile_path)
                        except Exception as e:
                            st.warning(f"Could not remove temporary file: {e}")


# Add custom CSS to the page
st.markdown(
    """
    <style>
    #indicator-descriptions-and-usage {
        padding-top: 14rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a note at the bottom of the page
st.subheader("Indicator Descriptions and Usage")

st.markdown("""
***
### Simple Moving Average (SMA)
- **Description**: The SMA is the average stock price over a specific period. It smooths out price data to identify trends.
- **Usage**: A rising SMA indicates an uptrend, while a falling SMA indicates a downtrend. Crossovers between short-term and long-term SMAs can signal buy/sell opportunities.
- **Best Combination**: Use with **RSI** to confirm overbought/oversold conditions during trend identification.

### Exponential Moving Average (EMA)
- **Description**: The EMA gives more weight to recent prices, making it more responsive to new information.
- **Usage**: Similar to SMA, but reacts faster to price changes. Useful for identifying short-term trends.
- **Best Combination**: Combine with **MACD** to confirm trend direction and momentum.

### Bollinger Bands
- **Description**: Bollinger Bands consist of a middle band (SMA) and two outer bands (standard deviations away from the SMA).
- **Usage**: Used to measure volatility. Prices near the upper band may indicate overbought conditions, while prices near the lower band may indicate oversold conditions.
- **Best Combination**: Use with **RSI** to confirm overbought/oversold conditions when prices touch the bands.

### Volume Weighted Average Price (VWAP)
- **Description**: VWAP is the average price a stock has traded at throughout the day, based on both volume and price.
- **Usage**: Often used by institutional traders to ensure they are getting a fair price. Prices above VWAP may indicate bullish sentiment, while prices below may indicate bearish sentiment.
- **Best Combination**: Combine with **OBV** to confirm volume trends alongside price trends.

### Relative Strength Index (RSI)
- **Description**: RSI measures the speed and change of price movements, ranging from 0 to 100.
- **Usage**: An RSI above 70 indicates overbought conditions, while an RSI below 30 indicates oversold conditions.
- **Best Combination**: Use with **MACD** to confirm momentum and trend reversals.

### Moving Average Convergence Divergence (MACD)
- **Description**: MACD is a trend-following momentum indicator that shows the relationship between two moving averages.
- **Usage**: A MACD crossover above the signal line may indicate a buy signal, while a crossover below may indicate a sell signal.
- **Best Combination**: Combine with **RSI** to confirm momentum and overbought/oversold conditions.

### Stochastic Oscillator
- **Description**: The Stochastic Oscillator compares a stock's closing price to its price range over a specific period.
- **Usage**: Values above 80 indicate overbought conditions, while values below 20 indicate oversold conditions.
- **Best Combination**: Use with **Bollinger Bands** to confirm overbought/oversold conditions when prices are near the bands.

### Average True Range (ATR)
- **Description**: ATR measures market volatility by decomposing the entire range of an asset price for that period.
- **Usage**: Higher ATR values indicate higher volatility, which can be useful for setting stop-loss levels.
- **Best Combination**: Combine with **Parabolic SAR** to set dynamic stop-loss levels based on volatility.

### Ichimoku Cloud
- **Description**: The Ichimoku Cloud is a comprehensive indicator that provides information about support/resistance, trend direction, momentum, and trade signals.
- **Usage**: The cloud (Kumo) acts as support/resistance. Price above the cloud indicates an uptrend, while price below indicates a downtrend.
- **Best Combination**: Use with **RSI** to confirm overbought/oversold conditions within the trend identified by the cloud.

### Fibonacci Retracement
- **Description**: Fibonacci Retracement levels are horizontal lines that indicate where support and resistance are likely to occur.
- **Usage**: Traders use these levels to identify potential reversal points in the price.
- **Best Combination**: Combine with **RSI** or **Stochastic Oscillator** to confirm reversals at key Fibonacci levels.

### Parabolic SAR
- **Description**: The Parabolic SAR is a trend-following indicator that provides potential entry and exit points.
- **Usage**: When the dots are below the price, it indicates an uptrend. When the dots are above the price, it indicates a downtrend.
- **Best Combination**: Use with **ATR** to set stop-loss levels based on volatility.

### On-Balance Volume (OBV)
- **Description**: OBV measures buying and selling pressure by adding volume on up days and subtracting volume on down days.
- **Usage**: Rising OBV indicates buying pressure, while falling OBV indicates selling pressure. It can be used to confirm price trends.
- **Best Combination**: Combine with **VWAP** to confirm volume trends alongside price trends.
""")