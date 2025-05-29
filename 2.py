import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.io as pio

from xgboost import XGBRegressor
import ta  # pip install ta (thư viện chỉ báo kỹ thuật)

app = Flask(__name__)

DATA_PATH = r"D:\New folder\stock_market_data\nasdaq\csv"

def get_tickers():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv')]
    return [f.replace('.csv', '') for f in files]

tickers = get_tickers()

def load_data(ticker):
    file = os.path.join(DATA_PATH, f"{ticker}.csv")
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  # đọc ngày dd-mm-yyyy
    df.sort_values('Date', inplace=True)
    return df

def check_stock_active(df):
    cutoff_date = pd.Timestamp('2022-06-01')
    changes = df['Close'].diff().fillna(0) != 0
    last_change_idx = changes[::-1].idxmax()
    last_change_date = df.loc[last_change_idx, 'Date']
    if last_change_date < cutoff_date:
        return False, last_change_date.strftime('%d/%m/%Y')
    return True, None

def add_technical_indicators(df):
    df = df.copy()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    df['Price_Change'] = df['Close'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change()
    df = df.dropna()
    return df

def train_xgboost(df):
    df = add_technical_indicators(df)
    features = ['SMA_5', 'EMA_12', 'RSI_14', 'Price_Change', 'Volume_Change']
    X = df[features]
    y = df['Close'].shift(-1)  # Giá ngày kế tiếp
    X = X[:-1]
    y = y[:-1]
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X, y)
    return model, features

def forecast_xgboost(df, model, features):
    df_tech = add_technical_indicators(df)
    last_row = df_tech[features].iloc[[-1]]
    prediction = model.predict(last_row)[0]
    return prediction

@app.route("/", methods=["GET", "POST"])
def index():
    symbol = None
    plot_html = None
    error = None
    inactive_msg = None

    if request.method == "POST":
        symbol = request.form.get("symbol", "").upper()
        if symbol not in tickers:
            error = f"Cổ phiếu '{symbol}' không tồn tại."
        else:
            df = load_data(symbol)
            is_active, last_active_date = check_stock_active(df)
            if not is_active:
                inactive_msg = f"Cổ phiếu đã ngừng giao dịch từ ngày {last_active_date}."

            # Huấn luyện mô hình XGBoost và dự báo
            try:
                model, features = train_xgboost(df)
                forecast = forecast_xgboost(df, model, features)
            except Exception as e:
                error = f"Lỗi trong quá trình dự báo: {e}"
                forecast = df['Close'].iloc[-1]

            # Vẽ biểu đồ giá Close và vùng dự báo
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['Date'], y=df['Close'],
                mode='lines+markers',
                name='Giá đóng cửa'
            ))

            last_date = df['Date'].max()
            next_date = last_date + pd.Timedelta(days=1)
            low = forecast * 0.99
            high = forecast * 1.01

            fig.add_trace(go.Scatter(
                x=[last_date, next_date, next_date, last_date],
                y=[low, low, high, high],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(0,100,80,0)'),
                hoverinfo='skip',
                showlegend=True,
                name='Vùng dự báo'
            ))

            fig.add_trace(go.Scatter(
                x=[next_date],
                y=[forecast],
                mode='markers',
                marker=dict(color='green', size=12, symbol='triangle-up'),
                name='Dự báo'
            ))

            fig.update_layout(
                title=f"Giá cổ phiếu {symbol} và dự báo bước tiếp theo bằng XGBoost",
                xaxis_title='Ngày',
                yaxis_title='Giá đóng cửa',
                hovermode='x unified',
            )

            plot_html = pio.to_html(fig, full_html=False)

    return render_template("index.html", tickers=tickers, symbol=symbol, plot_html=plot_html, error=error, inactive_msg=inactive_msg)

@app.route("/autocomplete")
def autocomplete():
    q = request.args.get("q", "").upper()
    results = [t for t in tickers if t.startswith(q)]
    return jsonify(results[:10])

if __name__ == "__main__":
    app.run(debug=True)
