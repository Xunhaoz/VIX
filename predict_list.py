import train_lstm as tl

from tensorflow import keras

adj_close_df = tl.get_stock(['TSM', "^TWII"])
all_data_x, all_data_y, scalers = tl.preprocess(adj_close_df, rolling=60)

model = keras.models.load_model('500epochs.h5')
predictions = model.predict(all_data_x)

tl.result(predictions, all_data_y, scalers)