import yfinance as yf
import ta
import matplotlib.pyplot as plt
import numpy as np

# 定義 ATR 周期
atr_period = 10

# 定義進場和出場的 ATR 倍律
entry_multiplier = 1
exit_multiplier = 2

# 定義交易期間
start_date = '2000-01-11'
end_date = '2023-04-15'

# 獲取股票價格數據
symbol = '2330.TW'
data = yf.download(symbol, start=start_date, end=end_date, interval='1d')

# 計算 ATR
data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'], window=atr_period)

# 初始化變數
position = False
entry_stop_loss_price = 0
exit_stop_loss_price = 0
trades = []
accumulated_profit = 0

# 進行交易
for i in range(1, len(data)):
    if data['Close'][i] > exit_stop_loss_price:
        if position:
            profit = data['Close'][i] - exit_stop_loss_price
            trades.append(profit)
            accumulated_profit += profit
            position = False
            exit_stop_loss_price = 0

    if not position and data['Close'][i] < data['Close'][i - 1] - entry_multiplier * data['ATR'][i]:
        position = True
        entry_stop_loss_price = data['Close'][i] - entry_multiplier * data['ATR'][i]

    if position and data['Close'][i] > data['Close'][i - 1] + exit_multiplier * data['ATR'][i]:
        exit_stop_loss_price = data['Close'][i] - exit_multiplier * data['ATR'][i]

# 定義買入點和賣出點
buy_points = np.array(
    [i + 1 for i in range(1, len(data)) if data['Close'][i] < data['Close'][i - 1] - entry_multiplier * data['ATR'][i]])
sell_points = np.array(
    [i + 1 for i in range(1, len(data)) if data['Close'][i] > data['Close'][i - 1] + exit_multiplier * data['ATR'][i]])

# 計算交易次數和累積損益
trade_count = len(trades)
accumulated_profit = round(accumulated_profit, 2)

# 計算單次最大獲利和單次最大損失
max_profit = max(trades) if trades else 0
max_loss = min(trades) if trades else 0

# 計算盈虧比
profit_factor = abs(max_profit / max_loss) if max_loss != 0 else 0

# 計算勝率
winning_trades = [trade for trade in trades if trade > 0]
win_rate = len(winning_trades) / trade_count * 100 if trade_count > 0 else 0

# 繪製股價圖表
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Stock Price')

# 顯示買入點
for buy_point in buy_points:
    plt.scatter(data.index[buy_point], data['Close'][buy_point], color='green', marker='^', label='Buy')

# 顯示賣出點
for sell_point in sell_points:
    plt.scatter(data.index[sell_point], data['Close'][sell_point], color='red', marker='v', label='Sell')

plt.grid(True)

# 顯示圖表和結果
plt.show()

# 輸出結果
print('交易次數:', trade_count)
print('累積損益:', accumulated_profit)
print('單次最大獲利:', round(max_profit, 2))
print('單次最大損失:', round(max_loss, 2))
print('盈虧比:', round(profit_factor, 2))
print('勝率:', round(win_rate, 2), '%')
