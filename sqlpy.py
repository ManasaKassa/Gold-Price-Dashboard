import mysql.connector
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(page_title="Gold Price Prediction Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Predict Price"])

# ==============================
# Load Data from MySQL
# ==============================
@st.cache_data
def load_data():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Manasakassa123456@gmail.com",  # change if needed
            database="GoldPrice_setd"
        )

        query = """
        SELECT 
            g.price_date,
            g.close_price AS target,  
            g.open_price,
            g.high_price,
            g.low_price,
            g.volume,
            e.inflation_rate,
            e.interest_rate,
            e.usd_index,
            e.crude_oil_price,
            e.stock_market_index,
            CASE 
                WHEN ge.impact_level = 'High' THEN 2
                WHEN ge.impact_level = 'Medium' THEN 1
                ELSE 0
            END AS event_impact
        FROM GoldPrices g
        LEFT JOIN EconomicIndicators e
            ON g.price_date = e.indicator_date
        LEFT JOIN GlobalEvents ge
            ON g.price_date = ge.event_date
        ORDER BY g.price_date ASC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # Ensure price_date is datetime
        df['price_date'] = pd.to_datetime(df['price_date'])

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()  # return empty DataFrame if error

df = load_data()

# ==============================
# Check if data loaded
# ==============================
if df.empty:
    st.warning("No data loaded! Please check your MySQL database and query.")
else:
    # ==============================
    # Train Model
    # ==============================
    X = df.drop(columns=['target', 'price_date'])
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    @st.cache_data
    def train_model(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    latest_price = df['target'].iloc[-1]
    next_day_price = model.predict([X.iloc[-1]])[0]

    # ==============================
    # Dashboard Page
    # ==============================
    if page == "Dashboard":
        st.title("Gold Price Prediction Dashboard")

        # Filters
        st.sidebar.header("Filters")
        start_date = st.sidebar.date_input("Start Date", df["price_date"].min().date())
        end_date = st.sidebar.date_input("End Date", df["price_date"].max().date())

        # Convert to Timestamp for filtering
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)

        df_filtered = df[(df["price_date"] >= start_date_ts) & (df["price_date"] <= end_date_ts)].copy()

        if df_filtered.empty:
            st.warning("No data available for the selected date range!")
        else:
            # KPIs
            st.subheader("Key Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Latest Gold Price", f"{latest_price:.2f} USD")
            col2.metric("Predicted Next Price", f"{next_day_price:.2f} USD")
            col3.metric("Model MSE", f"{mse:.2f}")

            # Actual vs Predicted
            st.subheader("Actual vs Predicted Prices")
            fig1, ax1 = plt.subplots(figsize=(10,5))
            ax1.plot(y_test.values, label="Actual", color="blue")
            ax1.plot(y_pred, label="Predicted", color="red")
            ax1.legend()
            st.pyplot(fig1)

            # Historical Prices
            st.subheader("Historical Gold Prices")
            fig2, ax2 = plt.subplots(figsize=(10,5))
            ax2.plot(df_filtered["price_date"], df_filtered["target"], color="gold")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Gold Price (USD)")
            st.pyplot(fig2)

            # Feature Correlation Heatmap
            st.subheader("Feature Correlation Heatmap")
            fig3, ax3 = plt.subplots(figsize=(10,5))
            sns.heatmap(df.drop(columns=['price_date']).corr(), annot=True, cmap="coolwarm", ax=ax3)
            st.pyplot(fig3)

    # ==============================
    # Prediction Page
    # ==============================
    elif page == "Predict Price":
        st.title("Predict Gold Price")

        col1, col2 = st.columns(2)

        with col1:
            open_price = st.number_input("Open Price", min_value=0.0, value=float(df['open_price'].mean()))
            high_price = st.number_input("High Price", min_value=0.0, value=float(df['high_price'].mean()))
            low_price = st.number_input("Low Price", min_value=0.0, value=float(df['low_price'].mean()))
            volume = st.number_input("Volume", min_value=0.0, value=float(df['volume'].mean()))

        with col2:
            inflation_rate = st.number_input("Inflation Rate", value=float(df['inflation_rate'].mean()))
            interest_rate = st.number_input("Interest Rate", value=float(df['interest_rate'].mean()))
            usd_index = st.number_input("USD Index", value=float(df['usd_index'].mean()))
            crude_oil_price = st.number_input("Crude Oil Price", value=float(df['crude_oil_price'].mean()))
            stock_market_index = st.number_input("Stock Market Index", value=float(df['stock_market_index'].mean()))
            event_impact = st.selectbox("Event Impact", [0,1,2], format_func=lambda x: ["Low","Medium","High"][x])

        if st.button("Predict Price"):
            user_input = [[open_price, high_price, low_price, volume,
                           inflation_rate, interest_rate, usd_index,
                           crude_oil_price, stock_market_index, event_impact]]
            prediction = model.predict(user_input)[0]
            st.success(f"Predicted Gold Price: {prediction:.2f} USD")
