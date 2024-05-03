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
from buy_sell import open_buy_position, open_sell_position, close_position, get_balance
from tabulate import tabulate
from lpt_model import pred


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
symbol = "LPTUSDT"
interval = "1h"
limit = 1440
position_open_value = 0
buy_amount = 30
sell_amount = 30
start_trailing_at = 0.003 # 0.30%
stop_loss_trailing_range = 0.1 # 0.10%
# amount = 30
threshold = 0.5
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


def preproccess_data(df):

    data_next_open = []
    for i in range(df.shape[0]):
        if i != df.shape[0]-1:
            data_next_open.append(df["open"][i+1])
        else:
            df.drop(df.tail(1).index,inplace=True)
            df.insert(11, value=data_next_open, column='next_open')
            df.dropna(axis=1, inplace=True)
            df.drop(columns='volume', inplace=True)
            df.drop(columns='taker_buy_base_asset_volume', inplace=True)
            df.drop(columns='ignore', inplace=True)

    return df


def fetch_binance_data(symbol, interval):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": 2
    }
    response = requests.get(url, params=params)
    result = response.json()

    df = pd.DataFrame(result, columns=[
        "timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_asset_volume", "number_of_trades", "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume", "ignore"
    ])


    # Convert the appropriate columns to float
    numeric_columns = ["open", "high", "low", "close", "volume", "quote_asset_volume",
                       "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
    df[numeric_columns] = df[numeric_columns].astype(float)
    df.drop(columns='timestamp', inplace=True)

    time.sleep(1)  # Add a delay of 1 second between requests to avoid rate limiting

    return df

def log_to_csv(time, model_accuracy, open_val, high_val, low_val, current_price, model1_prediction, model2_prediction, trade_type, total_profit, amount, predicted_price_change, predicted_profit, profit, highest_profit, lowest_profit, trade_predeicted_time, trade_time):
    csv_log_file = 'log_file_model1_twoDecisionsLogged_trailing_lptTUSD.csv'
    with open(csv_log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time, model_accuracy, open_val, high_val, low_val, current_price, model1_prediction, model2_prediction, trade_type, total_profit, amount, predicted_price_change, predicted_profit, profit, highest_profit, lowest_profit, trade_predeicted_time, trade_time])

def calculate_profit1(previous_price, position, amount):
    """
    Calculate the profit based on the current price and the previous price.

    Parameters:
        current_price (float): The current price.
        previous_price (float): The previous price.
        position (str): The position ('long' for buy position, 'short' for sell position).

    Returns:
        float: The calculated profit.
    """
    global symbol, interval
    current_price = fetch_binance_data(symbol, interval)['close'][1]
    if position == 'long':
        # Long position (buy)
        profit = (current_price - previous_price)*(amount/current_price)
    elif position == 'short':
        # Short position (sell)
        profit = (previous_price - current_price)*(amount/current_price)
    else:
        raise ValueError("Invalid position. Please specify 'long' or 'short'.")

    return profit, current_price


def calculate_profit(previous_amount, position, trade_amount):
    """
    Calculate the profit based on the current price and the previous price.

    Parameters:
        current_price (float): The current price.
        previous_price (float): The previous price.
        position (str): The position ('long' for buy position, 'short' for sell position).

    Returns:
        float: The calculated profit.
    """



    global symbol, interval
    current_amount = get_balance(symbol, previous_amount)['total']
    profit = current_amount - previous_amount
    # + (0.08/100)*trade_amount
    current_price = fetch_binance_data(symbol, interval)['close'][1]


    return profit, current_price, profit/previous_amount

def print_table(open_val, high_val, low_val, current_price,pred, total_profit, amount, predicted_price_change, predicted_price_change_ammount):
    table_data = [
        ["open", "high", "low", "close", "pred", "Total Profit", "amount", "predicted_price_change", "predicted_price_change/ammount"],
        [f"{open_val}", f"{high_val}", f"{low_val}", f"{current_price}",pred, f"{total_profit}", f"{amount}", f"{predicted_price_change}", f"{predicted_price_change_ammount:3f}"]
    ]
    # [f"{open_val:.2f}", f"{high_val:.2f}", f"{low_val:.2f}", f"{current_price:.2f}",pred, f"{total_profit:.3f}", f"{amount:.3f}", f"{float(predicted_price_change):.4f}", f"{float(predicted_price_change_ammount):.4f}"]


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
    global threshold
    global start_trailing_at
    hours = 0
    minutes = 0

    # historical_data = fetch_historical_data()
    # print("historical has been prepared successfully.")
    # df_1000 = prepare_dataset(historical_data)
    # model, model_accuracy = train_model(df_1000)
    # print("model has been trained successfully.")



    while True:
        try:
            # pred1 = 2
            # current_volume, volume_before, previous_price, current_price = fetch_real_time_data1()
            #
            # buy = 0
            # sell = 0
            # wait = 0
            # decision = 2
            # for _ in range(1):
            #     predicted_price_change = predict_price_change(model, current_volume, volume_before, previous_price)
            #     if predicted_price_change > position_open_value:
            #         buy+=1
            #     elif predicted_price_change < -1 * position_open_value:
            #         sell+=1
            #     else:
            #         wait+=1
            #
            # if buy>sell and buy>wait:
            #     pred1 = 0
            # elif sell>buy and sell>wait:
            #     pred1 = 1
            # elif wait>buy and wait>sell:
            #     pred1 = 2
            # else:
            #     pred1 = 2


            # if int(minutes) == 30 or (int(minutes) == 0 and int(hours) != 0):
            #     historical_data = fetch_historical_data()
            #     print("historical has been prepared successfully.")
            #     df1000 = prepare_dataset(historical_data)
            #     df1000.head()
            #     model, model_accuracy = train_model(df1000)
            #     print("model2 has been trained successfully.")

            df = fetch_binance_data(symbol, '1h')
            print(df.head())
            df = preproccess_data(df)
            current_price = df['close'][0]
            high = df['high'][0]
            low = df['low'][0]
            open = df['open'][0]
            running_time = time.time() - start_time
            now = datetime.datetime.now()
            formatted_time = now.strftime("%d:%m:%Y %H:%M:%S")
            hours, remainder = divmod(running_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            running_time = time.time() - start_time
            hours, remainder = divmod(running_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            # position_open_value = 0.2 / (amount/current_price)
            # buy = 0
            # sell = 0
            # wait = 0
            # decision = 2
            # for _ in range(1):
            #     result = pred(df)
            #     if result == 0:
            #         buy+=1
            #     elif result == 1:
            #         sell+=1
            #     else:
            #         wait+=1
            #
            #
            # if buy>sell and buy>wait:
            #     decision = 0
            # elif sell>buy and sell>wait:
            #     decision = 1
            # elif wait>buy and wait>sell:
            #     decision = 2
            # else:
            #     decision = 2
            # print(df.columns)
            result = pred(df)
            print("result =>", result)
            predicted_change_pct = ((result - df['close'])/df['close'])*100
            predicted_change_pct = predicted_change_pct[0]
            print("predicted profit =>", predicted_change_pct)

            if predicted_change_pct > threshold:
                action = 'BUY'
            elif predicted_change_pct < -1 * threshold:
                action = 'SELL'
            else:
                action = 'WAIT'
            amount = get_balance(symbol, amount)
            previous_amount = amount['total']
            amount = amount['free']
            # if amount < risk_line:
            #     exit()
            print_table(open,high,low,current_price, action,total_profit,amount,predicted_change_pct, ((predicted_change_pct/100)*current_price)*(amount/current_price))

            highest_profit = float('-inf')
            lowest_profit = float('inf')
            trade_type = None
            stop_loss = -1
            trade_predeicted_time = 60*60-seconds

            if action == 'BUY':

                print("close after:", 60-seconds, "seconds")

                # delete counter when implementation
                counter+=1
                # Buy BTCUSDT
                buy_quantity = (amount/current_price) - 0.7
                buy_quantity = round(buy_quantity, 3)
                if buy_quantity > (amount/current_price):
                    buy_quantity -= 0.7
                buy_order = open_buy_position(symbol, buy_quantity)
                trade_start_time = time.time()
                if buy_order:
                    # Do something with the buy_order, e.g., logging or processing
                    print("Buy order details:", buy_order)

                sleep_time = 60                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           /90
                # take_profit = predicted_price_change/current_price
                take_profit = (threshold/100) * (buy_quantity*current_price)
                trade_run = True
                sec_counter = -1
                pass_permession = 0
                trailing = 0
                while trade_run:
                    sec_counter += 1
                    i = sec_counter
                    time.sleep(1)
                    # profit, close_price = calculate_profit(current_price, "long", buy_quantity*current_price)
                    profit, close_price, profit_percentage = calculate_profit(previous_amount, "long", buy_quantity*current_price)

                    # trade_time = sleep_time*(i+1)
                    trade_time = time.time() - trade_start_time
                    if trade_time >= (i+1)*60:
                        print("close price =", close_price)
                        print(f"profit in ({sleep_time*(i+1)}) seconds = ${profit} ({profit_percentage*100}%)")


                    if profit_percentage >= predicted_change_pct*0.9:
                        break

                    if profit_percentage > highest_profit:
                        highest_profit = profit_percentage

                    if profit_percentage < lowest_profit:
                        lowest_profit = profit_percentage

                    if profit_percentage >= start_trailing_at and not trailing:
                        print("---!Trailing Started!---")
                        trailing = 1

                    if trailing:
                        if profit_percentage <= highest_profit - stop_loss_trailing_range/100:
                            break



                    if trade_time >= (60*60-seconds):
                        break

                    # if (trade_time >= (60*60-seconds)) or pass_permession or profit_percentage>=start_trailing_at:
                    #     pass_permession = 1
                    #     print("--Trailing Started--")
                    #     if profit >= take_profit:
                    #         print("--Trailing Started--")
                    #         stop_loss = profit - stop_loss_trailing_range * (buy_quantity*current_price)
                    #         take_profit = profit
                    #
                    #     if (profit <= 0 and (trade_time >= (60*60-seconds))) or (profit <= stop_loss):
                    #         break
                    #         # df = fetch_real_time_data2()
                    #         # result = pred(df)
                    #         # if result == 0:
                    #         #     continue
                    #         # else:
                    #         #     break
                    #     if trade_time >= (60*60-seconds) and profit_percentage <=0.3:
                    #         break


                # close position
                close_order = close_position(symbol, buy_quantity, 'sell')
                close_price = fetch_real_time_data2()['close'][1]
                fees = (threshold/100) * (buy_quantity*current_price)
                profit = (close_price - current_price) * buy_quantity - fees
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

                if (trade_time < (60*60-seconds)):
                    print(f"WAITING FOR THE NEXT HOUR TO START ({(60*60-seconds) - trade_time} seconds remains)")
                    time.sleep((60*60-seconds) - trade_time)

            elif action == "SELL":
                print("close after:", 60*60-seconds, "seconds")
                trade_type = "SELL"
                # delete counter when implementation
                counter+=1
                # SELL BTCUSDT
                sell_quantity = (amount/current_price)
                sell_quantity = round(sell_quantity, 3) - 0.7
                if sell_quantity > (amount/current_price):
                    sell_quantity -= 0.7
                print(sell_quantity)
                # Open a sell position
                sell_order = open_sell_position(symbol, sell_quantity)  # Example sell price
                print("Sell Order:", sell_order)
                trade_start_time = time.time()
                if sell_order:
                    # Do something with the buy_order, e.g., logging or processing
                    print("SELL order details:", sell_order)

                # 50/30,000 => 0.001
                sleep_time = 60

                # take_profit = -1*(predicted_price_change/current_price)
                take_profit = (threshold/100) * (sell_quantity*current_price)
                trade_run = True
                sec_counter = -1
                pass_permession = 0
                trailing = 0
                while trade_run:
                    sec_counter += 1
                    i = sec_counter
                    time.sleep(1)
                    # profit, close_price = calculate_profit(current_price, "short", sell_quantity*current_price)
                    profit, close_price, profit_percentage = calculate_profit(previous_amount, "short", sell_quantity*current_price)
                    # trade_time = sleep_time*(i+1)
                    trade_time = time.time() - trade_start_time

                    if trade_time%60 <= 4:
                        print("close price =", close_price)
                        print(f"profit in ({sleep_time*(i+1)}) seconds = ${profit} ({profit_percentage*100}%)")


                    if profit_percentage > highest_profit:
                        highest_profit = profit_percentage

                    if profit_percentage < lowest_profit:
                        lowest_profit = profit_percentage


                    if profit_percentage >= start_trailing_at and not trailing:
                        print("---!Trailing Started!---")
                        trailing = 1

                    if trailing:
                        if profit_percentage <= highest_profit - stop_loss_trailing_range/100:
                            break


                    if profit_percentage >= -1*predicted_change_pct*0.9:
                        break

                    if trade_time >= (60*60-seconds):
                        break
                    # if trade_time >= (60*60-seconds) or pass_permession or profit_percentage>=start_trailing_at:
                    #     # or profit >= ((-predicted_change_pct/100)*current_price)*(amount/current_price)
                    #     print("--Trailing Started--")
                    #     pass_permession = 0
                    #     if profit >= take_profit:
                    #         print("--Trailing Started--")
                    #         stop_loss = profit - stop_loss_trailing_range*(sell_quantity*current_price)
                    #         take_profit = profit
                    #
                    #     if (profit <= 0 and (trade_time >= (60*60-seconds))) or (profit <= stop_loss):
                    #         break
                    #         # df = fetch_real_time_data2()
                    #         # result = pred(df)
                    #         # if result == 1:
                    #         #     continue
                    #         # else:
                    #         #     break
                    #
                    #     if trade_time >= (60*60-seconds) and profit_percentage <=0.3:
                    #         break

                # close position
                close_order = close_position(symbol, sell_quantity, 'buy')

                close_price = fetch_real_time_data2()['close'][1]
                fees = (threshold/100) * amount
                profit = -1*(close_price - current_price) * sell_quantity - fees
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
                if (trade_time < (60*60-seconds)):
                    print(f"WAITING FOR THE NEXT HOUR TO START ({(60*60-seconds) - trade_time} seconds remains)")
                    time.sleep((60*60-seconds) - trade_time)



            else:
                close_price = -1
                profit = -1
                highest_profit = -1
                lowest_profit = -1
                time.sleep(60 * 10)



            if close_price != -1:
                try:
                    log_to_csv(formatted_time, 99.9, open, high, low, current_price, 0, action, total_profit, amount-profit, predicted_change_pct, predicted_change_pct*((amount-profit)/current_price),trade_type, profit, highest_profit, lowest_profit, trade_predeicted_time, trade_time)
                except PermissionError:
                    print("!!!Error logging the previous trade!!!")



            time.sleep(1)  # Wait for 1 second before fetching data again
        except requests.exceptions.ConnectionError:
            print("Connection Error")
            time.sleep(1)


# buy_amount = round(31.58700565/6.79, 3)
# buy_price = 40000  # Replace with the desired buy price
# amount = get_balance(symbol, buy_amount)['used']
# buy_amount = round(amount/6.99, 3)
#
#

# # # Open a buy position
# buy_order = open_buy_position(symbol, buy_amount)
# print("Buy Order:", buy_order)
# Open a sell position
# sell_order = open_sell_position(symbol, sell_amount)  # Example sell price
# print("Sell Order:", sell_order)
#
amount = get_balance(symbol, buy_amount)['free']
risk_line = amount*0.90



# Close the position

# close_order = close_position(symbol, sell_amount, 'buy')
# print("Close Order:", close_order)
#
# buy_order = open_buy_position(symbol, buy_amount)  # Example sell price
# print("Buy Order:", buy_order)

# Close the position
# close_order = close_position(symbol, sell_amount, 'sell')
# print("Close Order:", close_order)

# Run the bot
bot_loop()
