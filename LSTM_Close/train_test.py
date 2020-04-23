# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:06:53 2020

@author: lucas
"""

import stock_predictor as sp

sym = 'MMM'
batch_size = 20
time_steps = 60
lr = .0005
epochs = 50

lstm, hist, min_max, x_test_t, y_test_t = sp.lstm_model(batch_size, time_steps, sym, lr, epochs )

#lstm = load_model("TSLA_best_model.h5")

y_pred = lstm.predict(sp.trim_to_batch(x_test_t, batch_size), batch_size = batch_size)
y_pred = y_pred.flatten()
y_test_t = sp.trim_to_batch(y_test_t, batch_size)
y_pred_orig = (y_pred * min_max.data_range_[3]) + min_max.data_min_[3]
y_test_t_orig = (y_test_t * min_max.data_range_[3]) + min_max.data_min_[3]

sp.plot_prediction(y_pred_orig, y_test_t_orig, sym)
lstm.save(sym + '_model.h5')
sp.plot_loss(hist)


