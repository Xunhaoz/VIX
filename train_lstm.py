import utils

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, LSTM


def get_stock(stocks_name: list) -> pd.DataFrame:
    """
    :param stocks_name: list
    :return dataframe:  pd.DataFrame

    get stock without nan and multicolumn,
    - how to deal with nan
        - find the first none nan index, drop data before it
        - interpolate all nan inside the data
    """

    df = utils.get_stock(stocks_name)
    df = df[['Adj Close']]
    df.columns = df.columns.droplevel(0)

    return df


def preprocess(df: pd.DataFrame, rolling: int = 60) -> tuple[np.array, np.array, dict[str, MinMaxScaler]]:
    """

    :param df:  pd.DataFrame
    :param rolling: int
    :return all_data_x, all_data_y, scalers:tuple[np.array, np.array, dict[str, MinMaxScaler]]

    - how to deal with multicolumn
        - only leaves the Adj Close and flatten all stock name

    - how to deal with scalar
        - min max scalar

    - how to deal with spliting
        - according to rolling window size, x with be data in window size expect the last one, y will the last one
    """
    non_na_loc = df.dropna().index[0]
    df = df.loc[non_na_loc:]
    df = df.interpolate()

    scalers = {}
    for column_name, column_data in df.iteritems():
        scaler = MinMaxScaler()
        scaler_value = scaler.fit_transform(column_data.values.reshape(-1, 1))
        df[column_name] = scaler_value
        scalers[column_name] = scaler

    all_data_x = []
    all_data_y = []
    for i in range(60, len(df)):
        all_data_x.append(df.iloc[i - 60:i])
        all_data_y.append(df.iloc[i, 1])

    return np.array(all_data_x), np.array(all_data_y), scalers


def get_model(time_steps, feature_num):
    model = Sequential(name="lstm_for_TWII")
    model.add(LSTM(units=128, return_sequences=True, input_shape=(time_steps, feature_num), dropout=0.2,
                   activation='sigmoid'))
    model.add(TimeDistributed(Dense(3)))
    model.add(Flatten())
    model.add(Dense(units=1))
    return model


def result(predictions: np.array, all_data_y: np.array, scalers: dict[str, MinMaxScaler]) -> None:
    predictions = scalers["^TWII"].inverse_transform(predictions.reshape(-1, 1))
    all_y = scalers["^TWII"].inverse_transform(all_data_y.reshape(-1, 1))
    result = pd.DataFrame({"Predictions": predictions.flatten(), "Original": all_y.flatten()})
    result.to_csv("result.csv")
    utils.plot(result, graph_name="Result")


if __name__ == "__main__":
    adj_close_df = get_stock(['TSM', "^TWII"])
    all_data_x, all_data_y, scalers = preprocess(adj_close_df, rolling=60)

    X_train, X_test, y_train, y_test = train_test_split(all_data_x, all_data_y, test_size=0.3, random_state=42)

    model = get_model(all_data_x.shape[1], all_data_x.shape[2])
    model.compile(loss='mean_squared_error', optimizer='adam')
    history_data = model.fit(X_train, y_train, epochs=500, batch_size=2048, verbose=1, validation_data=(X_test, y_test))
    model.save('500epochs.h5')

    loss = history_data.history['loss']
    val_loss = history_data.history['val_loss']
    history = pd.DataFrame({"loss": loss, "val_loss": val_loss})
    history.to_csv("history.csv")

    predictions = model.predict(all_data_x)
    result(predictions, all_data_y, scalers)
