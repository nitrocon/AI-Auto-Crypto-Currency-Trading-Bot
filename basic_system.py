import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import time
import warnings
from test import pred

warnings.filterwarnings("ignore")

total_profit = 0
counter = 0
profits = []
# Binance API credentials
api_key = "placeholder"
api_secret = "placeholder"

# Binance API endpoint URLs
base_url = "https://api.binance.com"
klines_url = base_url + "/api/v3/klines"

# Symbol and parameters
symbol = "BTCUSDT"
interval = "1m"
limit = 1440
position_open_value = 2

amount = 100
start_time = time.time()
# Initialize Binance API session
session = requests.Session()
session.headers.update({'Accept': 'application/json', 'User-Agent': 'Mozilla/5.0'})

# Fetch historical data from Binance API
def fetch_historical_data():
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    response = session.get(klines_url, params=params)
    data = response.json()
    df = pd.DataFrame(data)
    df = df.iloc[:, :6]  # Extract only the OHLCV columns
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df

# Prepare the dataset
def prepare_dataset(df):
    df["volume_before"] = df["volume"].shift(1)
    df["previous_price"] = df["close"].shift(1)
    df["price_change"] = df["close"].shift(-1) - df["close"]
    df = df.dropna()
    return df

# Train the model
def train_model(df):
    accuracy = 0

    X = df[["volume", "volume_before", "previous_price"]]
    y = df["price_change"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Create a pipeline with feature scaling and ensemble model
    pipeline = make_pipeline(StandardScaler(), GradientBoostingRegressor())

    # Define the hyperparameters to search over
    param_grid = {
        "gradientboostingregressor__n_estimators": [100, 200, 300],
        "gradientboostingregressor__learning_rate": [0.1, 0.05, 0.01],
        "gradientboostingregressor__max_depth": [3, 4, 5]
    }

    # Perform grid search to find the best combination of hyperparameters
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate the model on the test data
    accuracy = evaluate_model(best_model, X_test, y_test)
    print("Accuracy:", accuracy)

    return best_model, accuracy

def evaluate_model(model, X_test, y_test):
    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy
    acc = []
    for i in range(len(y_pred)):
        if y_pred[i] > position_open_value and y_test.values[i] > 0:
            acc.append(1)
        elif y_pred[i] < -1*position_open_value and y_test.values[i] < 0:
            acc.append(1)
        elif y_pred[i] < -1*position_open_value and y_test.values[i] > 0:
            acc.append(0)
        elif y_pred[i] > 1*position_open_value and y_test.values[i] < 0:
            acc.append(0)
        else:
            acc.append(1)

    accuracy = sum(acc) / len(acc)
    print("Model Accuracy:", accuracy)
    return accuracy


# Predict price change
def predict_price_change(model, current_volume, volume_before, previous_price):
    X = [[current_volume, volume_before, previous_price]]
    predicted_price_change = model.predict(X)
    return predicted_price_change

# Fetch real-time volume data from Binance API
def fetch_real_time_data():
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 2
    }
    response = session.get(klines_url, params=params)
    data = response.json()

    df_data = {
        "open": [float(entry[1]) for entry in data],
        "high": [float(entry[2]) for entry in data],
        "low": [float(entry[3]) for entry in data],
        "close": [float(entry[4]) for entry in data],
        "volume": [float(entry[5]) for entry in data]
    }

    df = pd.DataFrame(df_data)

    return df





def log_prices(predicted, last, real):
    if last > real:
        event = "SELL"
    elif real > last:
        event = "BUY"
    else:
        event = "WAIT"
    if real == 0:
        event = "-"


    line = f"{last},{real},{last-real},{event},{predicted}\n"  # Create the line to be appended

    with open("prediction_log.csv", "a") as file:
        file.write(line)  # Append the line to the file

# Main bot loop
def bot_loop():
    # normal bot without history saving
    global total_profit
    global counter
    global profits
    global buy_amount
    global sell_amount
    global amount




    while True:

        df = fetch_real_time_data()
        current_price = df['close'][1]
        volume = df['volume'][1]
        high = df['high'][1]
        low = df['low'][1]
        open = df['open'][1]
        running_time = time.time() - start_time
        hours, remainder = divmod(running_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        running_time = time.time() - start_time
        hours, remainder = divmod(running_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        predicted_price_change = pred(df)
        print(f"open        high        low          close        volume     pred    Total Profit       Amount\n{open}  {high}     {low}     {current_price}     {volume}       {predicted_price_change}        {total_profit}              {amount}")




        if predicted_price_change == 0:
            print("close after:", 60-seconds, "seconds")
            # delete counter when implementation
            counter+=1
            # Buy 0.001% of BTCUSDT
            buy_quantity = amount/current_price

            # Close position after 1 minute
            time.sleep(60-seconds)
            close_price = fetch_real_time_data()['close'][1]
            profit = (close_price - current_price) * buy_quantity
            print("Bought at:", current_price)
            print("Sold at:", close_price)
            print("Profit:", profit)
            total_profit += profit
            amount+= profit
            profits.append(profit)
            print(f"Total Profit = ${total_profit}")

            print(f"Running Time = {int(hours)}:{int(minutes)}:{int(seconds)}")

            print(profits)

        elif predicted_price_change == 1:
            print("close after:", 60-seconds, "seconds")
            # delete counter when implementation
            counter+=1
            # Buy 0.001% of BTCUSDT
            sell_quantity = amount/current_price
            # Close position after 1 minute
            time.sleep(60-seconds)
            close_price = fetch_real_time_data()['close'][1]
            profit = -1*(close_price - current_price) * sell_quantity
            print("Bought at:", current_price)
            print("Sold at:", close_price)
            print("Profit:", profit)

            total_profit += profit
            amount += profit
            profits.append(profit)
            print(f"Total Profit = ${total_profit}")

            print(f"Running Time = {int(hours)}:{int(minutes)}:{int(seconds)}")
            print(profits)

        else:
            close_price = 0

        log_prices(predicted_price_change, current_price, close_price)

        time.sleep(1)  # Wait for 1 second before fetching data again


# Run the bot
bot_loop()


