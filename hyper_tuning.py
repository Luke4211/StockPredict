# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 03:44:53 2020

@author: lucas
"""
import talos as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import stock_predictor as sp
import tensorflow.keras
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import load_model
from keras.callbacks import Callback
from keras.callbacks.callbacks import CSVLogger
import time
import pickle
import numpy as np

cols = ['open','high','low','close', 'volume']
mat = sp.get_data('GE').loc[:,cols].values
mat = np.asarray(mat, dtype='float64')
#mat = sp.get_train_test(mat)
def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")

csv_logger = CSVLogger('log_' + get_readable_ctime() + '.log', append=True)
class LogMetrics(Callback):

    def __init__(self, search_params, param, comb_no):
        self.param = param
        self.self_params = search_params
        self.comb_no = comb_no

    def on_epoch_end(self, epoch, logs):
        for i, key in enumerate(self.self_params.keys()):
            logs[key] = self.param[key]
        logs["combination_number"] = self.comb_no



search_params = {
    "lstm_layers": [1,2],
    "dense_layers": [1,2],
    "lstm1_nodes" : [70, 90, 100],
    "lstm2_nodes" : [40, 60, 70],
    "dense2_nodes" : [20, 30, 50],
    "batch_size": [20, 30, 40],
    "time_steps": [30, 60, 90],
    "lr": [0.001, 0.0001],
    "epochs": [50, 70, 100],
    "optimizer": ["rms", "adam"]
}

def data(search_params):
    
    global mat
    
    BATCH_SIZE = search_params["batch_size"]
    TIME_STEPS = search_params["time_steps"]
    
    x_train, x_test = train_test_split(mat, train_size=0.8, test_size=0.2, shuffle=False)

    # scale the train and test dataset
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)

    x_train_ts, y_train_ts =sp.get_timeseries(x_train, 3, TIME_STEPS)
    x_test_ts, y_test_ts = sp.get_timeseries(x_test, 3, TIME_STEPS)
    x_train_ts = sp.trim_to_batch(x_train_ts, BATCH_SIZE)
    y_train_ts = sp.trim_to_batch(y_train_ts, BATCH_SIZE)
    x_test_ts = sp.trim_to_batch(x_test_ts, BATCH_SIZE)
    y_test_ts = sp.trim_to_batch(y_test_ts, BATCH_SIZE)
    print("Test size(trimmed) {}, {}".format(x_test_ts.shape, y_test_ts.shape))
    return x_train_ts, y_train_ts, x_test_ts, y_test_ts

def create_model_talos(x_train_ts, y_train_ts, x_test_ts, y_test_ts, params):
    """
    function that builds model, trains, evaluates on validation data and returns Keras history object and model for
    talos scanning. Here I am creating data inside function because data preparation varies as per the selected value of 
    batch_size and time_steps during searching. So we ignore data that's received here as argument from scan method of Talos.
    """
    x_train_ts, y_train_ts, x_test_ts, y_test_ts = data(params)
    BATCH_SIZE = params["batch_size"]
    TIME_STEPS = params["time_steps"]
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(params["lstm1_nodes"], batch_input_shape=(BATCH_SIZE, TIME_STEPS, x_train_ts.shape[2]), dropout=0.2,
                        recurrent_dropout=0.2, stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))
    if params["lstm_layers"] == 2:
        lstm_model.add(LSTM(params["lstm2_nodes"], dropout=0.2))
    else:
        lstm_model.add(Flatten())

    if params["dense_layers"] == 2:
        lstm_model.add(Dense(params["dense2_nodes"], activation='relu'))

    lstm_model.add(Dense(1, activation='sigmoid'))
    if params["optimizer"] == 'rms':
        optimizer = optimizers.RMSprop(lr=params["lr"])
    else:
        optimizer = optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)  # binary_crossentropy
    history = lstm_model.fit(x_train_ts, y_train_ts, epochs=params["epochs"], verbose=2, batch_size=BATCH_SIZE,
                             validation_data=[x_test_ts, y_test_ts],
                             callbacks=[ csv_logger])
    return history, lstm_model
tmp = []
print("Starting Talos scanning...")
t = ta.Scan(x=mat,
            y=mat[:,0],
            model=create_model_talos,
            params=search_params,
            experiment_name='stock_ge',
            print_params = True)

pickle.dump(t, open("talos_res","wb"))

