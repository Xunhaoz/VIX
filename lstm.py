import utils
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, LSTM

import tensorflow as tf


def get_stock(stocks_name: list) -> tuple[pd.DataFrame, dict[str, MinMaxScaler]]:
    """
    :param stocks_name: list
    :return dataframe:  pd.DataFrame

    get stock without nan and multicolumn,
    - how to deal with nan
        - find the first none nan index, drop data before it
        - interpolate all nan inside the data

    - how to deal with multicolumn
        - only leaves the Adj Close and flatten all stock name

    - how to deal with scalar
        - min max scalar
    """

    df = utils.get_stock(stocks_name)
    df = df[['Adj Close']]
    df.columns = df.columns.droplevel(0)

    non_na_loc = df.dropna().index[0]
    df = df.loc[non_na_loc:]
    df = df.interpolate()

    scalers = {}
    for column_name, column_data in df.iteritems():
        scaler = MinMaxScaler()
        scaler_value = scaler.fit_transform(column_data.values.reshape(-1, 1))
        df[column_name] = scaler_value
        scalers[column_name] = scaler
    return df, scalers


def get_model(time_steps, feature_num):
    model = Sequential(name="lstm_for_TWII")
    model.add(LSTM(units=128, return_sequences=True, input_shape=(time_steps, feature_num), dropout=0.2,
                   activation='sigmoid'))
    model.add(TimeDistributed(Dense(3)))
    model.add(Flatten())
    model.add(Dense(units=1))
    return model



adj_close_df, scalers = get_stock(['TSM', "^TWII"])

all_data_x = []
all_data_y = []
for i in range(60, len(adj_close_df)):
    all_data_x.append(adj_close_df.iloc[i - 60:i])
    all_data_y.append(adj_close_df.iloc[i, 1])

all_data_x = np.array(all_data_x)
all_data_y = np.array(all_data_y)

X_train, X_test, y_train, y_test = train_test_split(all_data_x, all_data_y, test_size=0.3, random_state=42)

model = get_model(all_data_x.shape[1], all_data_x.shape[2])
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=500, batch_size=1024, verbose=1, validation_data=(X_test, y_test))

predictions = model.predict(all_data_x)
predictions = scalers["^TWII"].inverse_transform(predictions.reshape(-1, 1))
all_y = scalers["^TWII"].inverse_transform(all_data_y.reshape(-1, 1))
utils.plot(pd.DataFrame({"Predictions": predictions.flatten(), "Original": all_y.flatten()}), graph_name="Result")
