# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:06:48 2020

@author: lucas
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm_notebook
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers as op
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import load_model
from keras.callbacks.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
import talos as ta
# sym - Stock symbol. Example = IMB
# interval - interval, if daily=False. Example = 5min
# api_key - AlphaVantage API key
# daily - If true, grab by days
# Returns: a Pandas Dataframe containing stock data for prev 100 days
def get_data(sym, interval=None, api_key="alphavantage_api_key.txt", daily=True):
    f = open(api_key, 'r')
    key = f.read()
    url = ''
    if not daily:
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=' + sym + '&interval=' + interval + '&apikey=' + key +'&datatype=csv'
    else:
        url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&outputsize=full&symbol=' + sym + '&apikey=' + key + '&datatype=csv'
    return pd.read_csv(url, engine='python')

def data(mat, batch_size, steps):
    
    
    BATCH_SIZE = batch_size
    TIME_STEPS = steps
    
    x_train, x_test = train_test_split(mat, train_size=0.8, test_size=0.2, shuffle=False)

    # scale the train and test dataset
    min_max_scaler = MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)

    x_train_ts, y_train_ts =get_timeseries(x_train, 3, TIME_STEPS)
    x_test_ts, y_test_ts = get_timeseries(x_test, 3, TIME_STEPS)
    x_train_ts = trim_to_batch(x_train_ts, BATCH_SIZE)
    y_train_ts = trim_to_batch(y_train_ts, BATCH_SIZE)
    x_test_ts = trim_to_batch(x_test_ts, BATCH_SIZE)
    y_test_ts = trim_to_batch(y_test_ts, BATCH_SIZE)
    print("Test size(trimmed) {}, {}".format(x_test_ts.shape, y_test_ts.shape))
    return x_train_ts, y_train_ts, x_test_ts, y_test_ts, min_max_scaler

def plot_true_data(df, sym, mult = 1.3, min_p = 5, max_p = 57):
    plt.figure()
    plt.plot(df["open"])
    plt.plot(df["high"])
    plt.plot(df["low"])
    plt.plot(df["close"])
    #plt.axis([df.shape[0], 0, min(df["low"])/mult, max(df["high"])*mult])
    plt.axis([df.shape[0], 0, min_p, max_p])
    plt.title(sym + ' stock price history')
    plt.ylabel('Price (USD)')
    plt.xlabel('Days')
    plt.legend(['Open','High','Low','Close'], loc='upper left')
    plt.show()
    
def get_train_test(df, cols = ['open','high','low','close', 'volume'] ):
    df_train, df_test = train_test_split(df, train_size=0.8, test_size = 0.2, shuffle=False)
    x = df_train.loc[:,cols]
    min_max = MinMaxScaler()
    x_train = min_max.fit_transform(x)
    x_test = min_max.transform(df_test.loc[:,cols])
    
    
    return x_train, x_test, min_max

# Convert stock data to time series, for use
# in LSTM     
def get_timeseries(df, y_index, time_steps):
    dim_0 = df.shape[0] - time_steps
    dim_1 = df.shape[1]
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))
    
    for i in tqdm_notebook(range(dim_0)):
        x[i] = df[i:time_steps+i]
        y[i] = df[time_steps+i, y_index]
    return x, y

def trim_to_batch(df, batch_size):
    no_drop = df.shape[0]%batch_size
    if(no_drop > 0):
        return df[:-no_drop]
    else:
        return df
    
def create_LSTM(df, batch_size, time_steps, features, sym, lr=0.01, epochs = 1000):
    x_train, x_test, min_max = get_train_test(df)
    x_t, y_t = get_timeseries(x_train, 4, time_steps)
    x_t = trim_to_batch(x_t, batch_size)
    y_t = trim_to_batch(y_t, batch_size)
    x_temp, y_temp = get_timeseries(x_test, 4, time_steps)
    x_val, x_test_t = np.split(trim_to_batch(x_temp, batch_size),2)
    y_val, y_test_t = np.split(trim_to_batch(y_temp, batch_size),2)
    
    lstm = Sequential()
    lstm.add(LSTM(70, 
                  batch_input_shape=(batch_size, time_steps, x_t.shape[2]),
                  dropout=0.2, recurrent_dropout=0.2, stateful = True, return_sequences=True,
                  kernel_initializer='random_uniform' ))
    lstm.add(Flatten())
    #lstm.add(Dropout(0.5))
    #lstm.add(Dense(20,activation='relu'))
    lstm.add(Dense(1,activation='sigmoid'))
    opt = op.RMSprop(lr=lr)
    lstm.compile(loss='mean_squared_error', optimizer=opt)
    
    csv_log = CSVLogger(sym + "_log.log", append=True)
    
    
    history = lstm.fit(x_t, y_t, epochs=epochs, verbose=2,
                       batch_size = batch_size,
                       validation_data=[x_val, y_val],
                       callbacks = [csv_log])
    return [lstm, history, min_max, x_test_t, y_test_t ]

def lstm_model(batch_size, time_steps, sym, lr, epochs, dropout=.3):
    cols = ['open','high','low','close', 'volume']
    mat = get_data(sym).loc[:,cols].values
    x_train_ts, y_train_ts, x_test_ts, y_test_ts, scaler = data(mat, batch_size, time_steps)
    lstm = Sequential()
    lstm.add(LSTM(70, 
                  batch_input_shape=(batch_size, time_steps, x_train_ts.shape[2]),
                  dropout=0.1, recurrent_dropout=dropout, stateful = True, 
                  kernel_initializer='random_uniform' ))
    lstm.add(Flatten())
    lstm.add(Dense(1,activation='sigmoid'))
    opt = op.RMSprop(lr=lr)
    lstm.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
    
    
    csv_log = CSVLogger(sym + "_log.log", append=True)
    early_stp = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15, min_delta=1)
    checkpoint = ModelCheckpoint( str(sym) + '_best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    
    history = lstm.fit(x_train_ts, y_train_ts, epochs=epochs, verbose=2,
                       batch_size = batch_size,
                       validation_data=[x_test_ts, y_test_ts],
                       callbacks = [csv_log, early_stp, checkpoint])
    
    return lstm, history, scaler, x_test_ts, y_test_ts

def plot_loss(history):
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
'''
df_ge = get_data('GE', '60min')
plot_true_data(df_ge, 'GE')
print(len(df_ge))
rev_scale = MinMaxScaler()
cols = ['open','high','low','close', 'volume']
df_ge = df_ge.loc[:,cols]
'''


#test = df_ge.loc[:,cols].values
#temp = df_ge.loc[:,cols].values


#x, y = get_timeseries(test, 3, 60)

#x = trim_to_batch(x, 20)

#lstm, hist, min_max, x_test_t, y_test_t = create_LSTM(df_ge, 20, 30, 100, "GE", lr=0.001, epochs=100)
'''
sym = 'TSLA'
batch_size = 20
time_steps = 60
lr = .0005
epochs = 50

lstm, hist, min_max, x_test_t, y_test_t = lstm_model(batch_size, time_steps, sym, lr, epochs )

#lstm = load_model("TSLA_best_model.h5")

y_pred = lstm.predict(trim_to_batch(x_test_t, batch_size), batch_size = batch_size)
y_pred = y_pred.flatten()
y_test_t = trim_to_batch(y_test_t, batch_size)
y_pred_orig = (y_pred * min_max.data_range_[3]) + min_max.data_min_[3]
y_test_t_orig = (y_test_t * min_max.data_range_[3]) + min_max.data_min_[3]

'''

def plot_prediction(y_pred_orig, y_test_t_orig, sym):
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(y_pred_orig)
    plt.plot(y_test_t_orig)
    ax.set_xlim(y_test_t_orig.shape[0], 0)
    #plt.axis([y_test_t_orig.shape[0], 0, 4, 15])
    plt.title('Prediction vs Real Stock Price for ' + sym)
    plt.ylabel('Price')
    plt.xlabel('Days')
    plt.legend(['Prediction', 'Real'], loc='upper left')
    plt.show()
'''
plot_prediction(y_pred_orig, y_test_t_orig, sym)
lstm.save(sym + '_model.h5')
plot_loss(hist)

'''

#rev_scale.min_, rev_scale.scale_ = min_max.min_[3],min_max.scale_[3]
#print(df_ge)
#print(df_ge.iloc[0,4])


'''
narr = df_ge.loc[:,'close'].to_numpy()
narr = narr.reshape(-1,1)
min_max.fit(narr)
#print(first)

pred = lstm.predict(x[0:60,], batch_size = 60)
pred = pred.flatten()



pred2 = rev_scale.inverse_transform(pred)
#a = min(df_ge["close"])
#b = max(df_ge["close"])

#sd = df_ge.iloc[:,4].values.std(ddof=1)
#mean = df_ge.iloc[:,4].values.mean()
#print(sd)
#print(mean)
for i in range(len(pred)):
    print(str(pred[i]) + ' ' + str(pred2[i]))

print(len(pred))

plot_loss(hist)
'''