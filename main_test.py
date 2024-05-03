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
import datetime
import csv
from buy_sell import open_buy_position, open_sell_position, close_position
from tabulate import tabulate
from model import pred


warnings.filterwarnings("ignore")

highest_profit = float('-inf')
lowest_profit = float('inf')
trade_type = None

total_profit = 0
counter = 0
profits = []
# Binance API credentials
api_key = "u4gyQyJFz59YesfarXMuyI9OE150fQzax2mCPxsFneXOfmSVKVvrIQVagHQtkMAB"
api_secret = "EoSj3EE6mUHWlnyiArc4OLDJg8wURfGeoQD1tPi2KT5qki4agbwtTKNtqURqMQjD"

# Binance API endpoint URLs
base_url = "https://api.binance.com"
klines_url = base_url + "/api/v3/klines"

# Symbol and parameters
symbol = "BTCUSDT"
interval = "1m"
limit = 1440
position_open_value = 0
buy_amount = 100
sell_amount = 100
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
def fetch_real_time_data1():
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 2
    }
    response = session.get(klines_url, params=params)
    data = response.json()
    latest_data = data[-1]
    current_volume = float(latest_data[5])
    volume_before = float(data[-2][5])
    current_price = float(latest_data[4])
    previous_price = float(data[-2][4])
    return current_volume, volume_before, previous_price, current_price


def fetch_real_time_data2():
    params = {
        "symbol": 'BTCUSDT',
        "interval": '30m',
        "limit": 1
    }
    response = session.get(klines_url, params=params)
    data = response.json()

    # Ensure that the API response data is a list
    if not isinstance(data, list):
        print("API response data is not a list.")
        return None

    # Extract the first entry from the response (since we're fetching only one)
    entry = data[0]

    # Extract values for the DataFrame
    open_val = float(entry[1])
    high_val = float(entry[2])
    low_val = float(entry[3])
    close_val = float(entry[4])
    volume_val = float(entry[5])

    # Create a DataFrame
    df_data = {
        "open": [open_val],
        "high": [high_val],
        "low": [low_val],
        "close": [close_val],
        "volume": [volume_val]
    }

    df = pd.DataFrame(df_data)

    return df

def fetch_real_time_data3(symbol, interval):
    params = {
        "symbol": "BTCUSDT",
        "interval": '30m',
        "limit": 2
    }
    response = session.get(klines_url, params=params)
    data = response.json()

    # Extract all columns from the API response
    column_names = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                    'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                    'taker_buy_quote_asset_volume', 'ignore']

    # Create a dictionary to store the column data
    df_data = {col: [float(entry[i]) for entry in data] for i, col in enumerate(column_names)}

    # Create a DataFrame using the extracted column data
    df = pd.DataFrame(df_data)

    return df


def log_to_csv(time, model_accuracy, open_val, high_val, low_val, current_price, volume, model1_prediction, model2_prediction, trade_type, total_profit, amount, predicted_price_change, predicted_profit, profit, highest_profit, lowest_profit, trade_predeicted_time, trade_time):
    csv_log_file = 'log_file_new_model_twoDecisionsLogged_trailing_BTCTUSD(1).csv'
    with open(csv_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time, model_accuracy, open_val, high_val, low_val, current_price, volume, model1_prediction, model2_prediction, trade_type, total_profit, amount, predicted_price_change, predicted_profit, profit, highest_profit, lowest_profit, trade_predeicted_time, trade_time])

def calculate_profit(previous_price, position, amount):
    """
    Calculate the profit based on the current price and the previous price.

    Parameters:
        current_price (float): The current price.
        previous_price (float): The previous price.
        position (str): The position ('long' for buy position, 'short' for sell position).

    Returns:
        float: The calculated profit.
    """
    current_price = fetch_real_time_data2()['close']
    if position == 'long':
        # Long position (buy)
        profit = (current_price - previous_price)*(amount/current_price)
    elif position == 'short':
        # Short position (sell)
        profit = (previous_price - current_price)*(amount/current_price)
    else:
        raise ValueError("Invalid position. Please specify 'long' or 'short'.")

    return profit, current_price


def print_table(open_val, high_val, low_val, current_price, volume,pred, total_profit, amount, predicted_price_change, predicted_price_change_ammount):
    table_data = [
        ["open", "high", "low", "close", "volume", "pred", "Total Profit", "amount", "predicted_price_change", "predicted_price_change/ammount"],
        [f"{open_val}", f"{high_val}", f"{low_val}", f"{current_price}", f"{volume}", pred, f"{total_profit}", f"{amount}", f"{predicted_price_change}", f"{predicted_price_change_ammount}"]
    ]

    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))



def bot_loop():
    # normal bot without history saving
    global total_profit
    global counter
    global profits
    global buy_amount
    global sell_amount
    global amount
    global position_open_value
    hours = 0
    minutes = 0




    while True:
        try:
            current_volume, volume_before, previous_price, current_price = fetch_real_time_data1()

            decision = 2





            df = fetch_real_time_data2()
            current_price = df['close']
            volume = df['volume']
            high = df['high']
            low = df['low']
            open = df['open']
            running_time = time.time() - start_time
            now = datetime.datetime.now()
            formatted_time = now.strftime("%d:%m:%Y %H:%M:%S")
            hours, remainder = divmod(running_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            running_time = time.time() - start_time
            hours, remainder = divmod(running_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            # position_open_value = 0.2 / (amount/current_price)
            buy = 0
            sell = 0
            wait = 0
            decision = 2
            for _ in range(1):
                result = pred(df)
                if result >= position_open_value:
                    buy+=1
                elif result <= -1*position_open_value:
                    sell+=1
                else:
                    wait+=1


            if buy>sell and buy>wait:
                decision = 0
            elif sell>buy and sell>wait:
                decision = 1
            elif wait>buy and wait>sell:
                decision = 2
            else:
                decision = 2



            print_table(open,high,low,current_price,volume, decision,total_profit,amount,result, result*(amount/current_price))

            highest_profit = float('-inf')
            lowest_profit = float('inf')
            trade_type = None
            stop_loss = -1
            trade_predeicted_time = 60-seconds

            if decision == 0:
                print("close after:", 60-seconds, "seconds")

                # delete counter when implementation
                counter+=1
                # Buy BTCUSDT
                buy_quantity = (amount/current_price)
                buy_quantity = round(buy_quantity, 3)
                # buy_order = open_buy_position(symbol, buy_quantity)

                # if buy_order:
                #     # Do something with the buy_order, e.g., logging or processing
                #     print("Buy order details:", buy_order)

                sleep_time = (90)/90
                # take_profit = predicted_price_change/current_price
                take_profit = result*buy_quantity
                trade_run = True
                sec_counter = -1
                while trade_run:
                    sec_counter += 1
                    i = sec_counter
                    time.sleep(60)
                    profit, close_price = calculate_profit(current_price, "long", amount)

                    print(f"profit in ({i}) minutes = ${profit}")
                    trade_time = sleep_time*(i+1)
                    if profit > highest_profit:
                        highest_profit = profit

                    if profit < lowest_profit:
                        lowest_profit = profit



                    if profit >= take_profit:
                        stop_loss = profit - 0.001
                        take_profit = profit

                    if i >= 30:
                        if (profit <= 0 and sleep_time*(i+1) >= 60) or (profit < -0.0001) or (profit <= stop_loss):
                            df = fetch_real_time_data2()
                            result = pred(df)
                            if result == 0:
                                continue
                            else:
                                break


                # close position
                # close_order = close_position(symbol, buy_amount, 'sell')
                close_price = fetch_real_time_data2()['close'][1]
                profit = (close_price - current_price) * buy_quantity
                print("Bought at:", current_price)
                print("Sold at:", close_price)
                print("Profit:", profit)
                # profitx = close_order['cost'] - buy_order['cost']
                # print("Profit:", profitx)
                total_profit += profit
                amount+= profit
                profits.append(profit)
                print(f"Total Profit = ${total_profit}")

                print(f"Running Time = {int(hours)}:{int(minutes)}:{int(seconds)}")

                print(profits)

            elif decision == 1:
                trade_start_time = time.time()
                print("close after:", 60*30-seconds, "seconds")
                trade_type = "SELL"
                # delete counter when implementation
                counter+=1
                # SELL BTCUSDT
                sell_quantity = (amount/current_price)
                sell_quantity = round(sell_quantity, 3)
                # Open a sell position
                # sell_order = open_sell_position(symbol, sell_quantity)  # Example sell price
                # print("Sell Order:", sell_order)
                # if sell_order:
                #     # Do something with the buy_order, e.g., logging or processing
                #     print("SELL order details:", sell_order)

                # 50/30,000 => 0.001
                sleep_time = (90)/90
                # take_profit = -1*(predicted_price_change/current_price)
                take_profit = -1* result * sell_quantity
                trade_run = True
                sec_counter = -1
                while trade_run:
                    sec_counter += 1
                    i = sec_counter
                    time.sleep(sleep_time)
                    profit, close_price = calculate_profit(current_price, "short", amount)

                    print(f"profit in ({sleep_time*(i+1)}) seconds = ${profit}")



                    if profit > highest_profit:
                        highest_profit = profit

                    if profit < lowest_profit:
                        lowest_profit = profit

                    if profit >= take_profit:
                            stop_loss = profit - 0.001
                            take_profit = profit

                    if i >= 30:
                        if (profit <= 0 and sleep_time*(i+1) >= 60) or (profit < -0.0001) or (profit <= stop_loss):
                            # df = fetch_real_time_data2()
                            # result = pred(df)
                            # if result == 1:
                            #     continue
                            # else:
                            break

                # close position
                # close_order = close_position(symbol, sell_quantity, 'buy')

                trade_time = time.time() - trade_start_time
                close_price = fetch_real_time_data2()['close'][1]
                profit = -1*(close_price - current_price) * sell_quantity
                print("Bought at:", current_price)
                print("Sold at:", close_price)
                print("Profit:", profit)

                # profitx = close_order['cost'] - sell_order['cost']
                # print("Profit:", profitx)

                total_profit += profit
                amount += profit
                profits.append(profit)
                print(f"Total Profit = ${total_profit}")

                print(f"Running Time = {int(hours)}:{int(minutes)}:{int(seconds)}")
                print(profits)


            else:
                close_price = -1
                profit = -1
                highest_profit = -1
                lowest_profit = -1


            if close_price != -1:
                try:
                    log_to_csv(formatted_time, -1, open, high, low, current_price, volume, decision, trade_type, total_profit, amount-profit, result, result*((amount-profit)/current_price),trade_type, profit, highest_profit, lowest_profit, trade_predeicted_time, trade_time)
                except PermissionError:
                    print("!!!Error logging the previous trade!!!")




            time.sleep(1)  # Wait for 1 second before fetching data again
        except requests.exceptions.ConnectionError:
            print("Connection Error")
            time.sleep(1)


# buy_amount = round(30/30000, 3)
# buy_price = 40000  # Replace with the desired buy price
# sell_amount = round(30/30000, 3)

# Open a buy position
# buy_order = open_buy_position(symbol, buy_amount)
# print("Buy Order:", buy_order)

# Open a sell position
# sell_order = open_sell_position(symbol, sell_amount)  # Example sell price
# print("Sell Order:", sell_order)
#
# # Close the position
# close_order = close_position(symbol, sell_amount, 'buy')
# print("Close Order:", close_order)

# buy_order = open_buy_position(symbol, buy_amount)  # Example sell price
# print("Buy Order:", buy_order)
#
# # Close the position
# close_order = close_position(symbol, sell_amount, 'sell')
# print("Close Order:", close_order)
# # Run the bot
bot_loop()

