import yfinance as yf
import pandas as pd
import utils
import warnings

warnings.filterwarnings("ignore")

data = utils.get_stock(["AAPL"], "./stocks")
print(data.head())

data = utils.cal_ma(data, [10, 22, 66], "Adj Close")
print(data.head())

data = utils.cal_atr(data, [22, 66])
print(data.head())

data = utils.cal_max_drawdown(data, [22, 66])
print(data.head())

data = utils.cal_daily_max_loss(data)
print(data.head())

data.to_csv("check.csv")

utils.plot(data[["Adj Close", "10MA", "22MA", "66MA"]], "Adj Close with MA")
utils.plot(data[["22ATR", "66ATR"]], "ATR")
utils.plot(data[["High-Low", "High-PrevClose", "Low-PrevClose"]], "Factor of ATR")
utils.plot(data[["22_drawdown", "66_drawdown"]], "max drawdown")
utils.plot(data[["Close-Low"]], "daily max loss")
