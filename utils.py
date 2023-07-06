import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


def get_stock(stock_name: list, folder_path: str = "./stocks") -> pd.DataFrame:
    stock_data_path = f"./stocks/{'-'.join(stock_name)}.csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    stock_data = yf.download(" ".join(stock_name), start=None, end=None, actions=False, threads=True, ignore_tz=None,
                             group_by='column', auto_adjust=False, back_adjust=False, repair=False, keepna=False,
                             progress=True, period="max", show_errors=None, interval="1d", prepost=False,
                             proxy=None, rounding=False, timeout=10, session=None)
    stock_data.to_csv(stock_data_path)
    return stock_data


def cal_ma(df: pd.DataFrame, ma_period: list = None, column_name: str = "Adj Close") -> pd.DataFrame:
    print("cal_ma:")
    if ma_period is None:
        ma_period = [10, 22, 66]
    for ma in ma_period:
        feature_column_name = f"{ma}MA"
        df[feature_column_name] = df[column_name].rolling(ma).mean()
    return df


def cal_atr(df: pd.DataFrame, atr_period: list = None) -> pd.DataFrame:
    print("cal_atr:")
    if atr_period is None:
        atr_period = [22, 66]
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TrueRange'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)

    for atr in atr_period:
        feature_column_name = f"{atr}ATR"
        df[feature_column_name] = df['TrueRange'].rolling(atr).mean()
    return df


def cal_max_drawdown(df: pd.DataFrame, mdd_period: list = None) -> pd.DataFrame:
    print("cal_max_drawdown:")
    if mdd_period is None:
        mdd_period = [22, 66]

    df.reset_index(inplace=True, drop=False)
    df_dicts = df.to_dict('records')

    for i in range(len(df_dicts)):
        for dd_days in mdd_period:
            index = min(dd_days - 1, i)
            if i < dd_days:
                df_slice = df_dicts[0:i + 1]
            else:
                df_slice = df_dicts[i - (dd_days - 1):i + 1]
            df_slice[index][f'{dd_days}_cumulative_high'] = max(df_slice, key=lambda x: x['Adj Close'])['Adj Close']
            df_slice[index][f'{dd_days}_drawdown'] = df_slice[index]['Adj Close'] / df_slice[index][
                f'{dd_days}_cumulative_high'] - 1
            df_slice[index][f'{dd_days}_max_drawdown'] = min(df_slice, key=lambda x: x[f'{dd_days}_drawdown'])[
                f'{dd_days}_drawdown']
    df = pd.DataFrame(df_dicts)
    df.set_index("Date", inplace=True)
    df.to_csv("check.csv")
    return df


def cal_daily_max_loss(df: pd.DataFrame) -> pd.DataFrame:
    print("cal_daily_max_loss:")
    df['Close-Low'] = df['Close'] - df['Low']
    return df


def plot(df: pd.DataFrame, graph_name: str = "Untitle") -> None:
    df.plot(linewidth=0.5)
    plt.ylabel('Adj Close')
    plt.xlabel("Time")
    plt.title(graph_name)
    plt.savefig(f"./images/{graph_name}.png", dpi=600)
    plt.show()
