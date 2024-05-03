import ccxt

# Initialize the CCXT exchange instance
exchange = ccxt.binance({
    'apiKey': '',
    'secret': '',
    'enableRateLimit': True,
    'options': {
        'defaultType': 'future',  # Set the default trade type to 'future'
    }
})

def open_buy_position(symbol, amount):

    return f"But Test Completed"

def open_sell_position(symbol, amount):

    return f"Sell Test Completed"

def close_position(symbol, amount, type):

    return f"Close Test Completed"


def get_balance(symbol, amount):
    try:
        # Define order parameters
        order_params = {
            'symbol': symbol,
            'side': 'sell',
            'type': 'market',
            'amount': amount,
        }
        wallets = exchange.fetch_balance()
        print(wallets['USDT'])
        return wallets['USDT']
    except:
        pass

