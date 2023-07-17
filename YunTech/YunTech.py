import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Flatten, TimeDistributed, LSTM
from keras.layers import Dropout

import matplotlib.pyplot as plt

df = pd.read_csv('ann_lstmdata.csv')


def preprocess():
    inverse_scalar = None
    data_groups = [
        [], [], [], []
    ]

    for index, data in df.iterrows():
        data = data.tolist()
        for i in range(0, 12, 3):
            data_groups[i // 3].extend(data[i: i + 3])
        data_groups[3].append(data[12])

    for k, data_group in enumerate(data_groups):
        scalar = MinMaxScaler()
        data_group = np.array(data_group).reshape(-1, 1)
        data_group = scalar.fit_transform(data_group)
        if k == 3:
            inverse_scalar = scalar
        data_groups[k] = data_group

    all_x = []
    all_y = []

    # 三組人流合併 放入train_x
    for index in range(0, len(data_groups[0]), 3):
        tmp = []
        for i in range(3):
            tmp.append(data_groups[i][index: index + 3])
        tmp = np.array(tmp).reshape(3, 3)
        all_x.append(tmp)

    # 循序提出train_x加入車流資料 train_y加入車流資料
    for k, index in enumerate(range(0, len(data_groups[3]), 4)):
        tmp = np.array(data_groups[3][index: index + 3]).flatten()
        all_x[k] = np.vstack((all_x[k], tmp)).T
        all_y.append(data_groups[3][index + 3])

    return np.array(all_x), np.array(all_y), inverse_scalar


def get_model(time_steps=3, feature_num=4):
    model = Sequential(name="lstm")
    model.add(LSTM(units=128, return_sequences=True, input_shape=(time_steps, feature_num), dropout=0.2,
                   activation='sigmoid'))
    model.add(TimeDistributed(Dense(3)))
    model.add(Flatten())
    model.add(Dense(units=1))
    return model


def plot(df: pd.DataFrame, graph_name: str = "Untitle") -> None:
    df.plot(linewidth=0.5)
    plt.ylabel('value')
    plt.xlabel("stops")
    plt.title(graph_name)
    plt.show()


if __name__ == "__main__":
    all_x, all_y, inverse_scalar = preprocess()
    X_train, X_test, y_train, y_test = train_test_split(all_x, all_y, test_size=0.3, random_state=42)

    model = get_model(X_train[0].shape[0], X_train[0].shape[1])
    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(X_train, y_train, epochs=315, batch_size=256, verbose=1, validation_data=(X_test, y_test))
    scores = model.evaluate(X_test, y_test, verbose=1)
    model.summary()

    predictions = model.predict(all_x)
    predictions = inverse_scalar.inverse_transform(predictions)
    all_y = inverse_scalar.inverse_transform(all_y)
    plot(pd.DataFrame({"Predictions": predictions.flatten(), "Original": all_y.flatten()}).head(500), graph_name="Result")
