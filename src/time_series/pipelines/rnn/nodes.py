from typing import Dict, Tuple
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import RNNModel
from darts.dataprocessing.transformers import Scaler
import pickle
import logging
import time
import tracemalloc
import torch


def fit_predict_rnn(train: pd.DataFrame, fit_rnn_p: Dict, complexity: Dict, predict: Dict) -> Dict:

    logger = logging.getLogger(__name__)

    ## Ejecución iterativa ('n_trials') para calcular tiempos y uso de memoria
    # El modelo que devuelve es el que se guarda en la última iteración
    fit_total_times = []
    fit_memory_peaks = []

    for trial in range(complexity['n_trials']):

        logger.info(f'Ajuste {trial+1} modelo')

        start_time = time.process_time()  # time.process_time()  time.perf_counter()
        tracemalloc.start()

        # Diferenciación
        train_diff = train.diff()
        train_diff.iloc[0, :] = train.iloc[0, :].copy()

        train_all_diff = [TimeSeries.from_dataframe(train_diff.loc[:, [f'y{i}']]) for i in range (1, len(train_diff.columns) + 1)]

        # Escalado
        scaler = Scaler()
        train_all_diff_scaled = scaler.fit_transform(train_all_diff)

        # Ajuste modelo
        rnn_model = RNNModel(input_chunk_length = fit_rnn_p['input_chunk_length'],
                            model = 'RNN',
                            hidden_dim = fit_rnn_p['hidden_dim'],
                            n_rnn_layers = fit_rnn_p['n_rnn_layers'],
                            n_epochs = fit_rnn_p['n_epochs'],
                            random_state = 123)
        rnn_model.fit(train_all_diff_scaled)   

        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        fit_total_times.append(end_time - start_time)
        fit_memory_peaks.append(mem_peak / 10**6)

    fit_avg_total_time = np.mean(fit_total_times)
    fit_total_obs = train.shape[0] * train.shape[1]
    # avg_series_time = np.mean(total_times) / train.shape[1]
    # series_obs = train.shape[0]
    fit_memory_peak = np.max(fit_memory_peaks)

    # Ejecución iterativa ('n_trials') para calcular tiempos y uso de memoria
    # La predicción que devuelve es la que se guarda en la última iteración
    pred_n_series_per_sec = []
    pred_memory_peaks = []

    for trial in range(complexity['n_trials']):

        logger.info(f'Predicción {trial+1}')

        start_time = time.process_time()  # time.process_time()  time.perf_counter()
        tracemalloc.start()

        # Predicción
        rnn_forecast_diff_scaled = rnn_model.predict(n = predict['n_periods'], series = train_all_diff_scaled)

        # Desescalado
        rnn_forecast_diff = scaler.inverse_transform(rnn_forecast_diff_scaled)

        rnn_forecast_diff_list = [rnn_forecast_diff[i].pd_dataframe() for i in range(len(rnn_forecast_diff))]
        rnn_forecast_diff = pd.concat(rnn_forecast_diff_list, axis = 1)

        # Integración = inversa diferenciación
        rnn_forecast = rnn_forecast_diff.copy()
        rnn_forecast.iloc[0, :] = rnn_forecast.iloc[0, :] + train.iloc[-1, :]
        rnn_forecast = rnn_forecast.cumsum(axis = 0) 

        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        pred_n_series_per_sec.append(len(train.columns) / (end_time - start_time))
        pred_memory_peaks.append(mem_peak / 10**6)
    
    pred_avg_series_per_sec = np.mean(pred_n_series_per_sec)
    pred_memory_peak = np.max(pred_memory_peaks)

    rnn_model = {'rnn_model': pickle.dumps(rnn_model)}

    return {
        'rnn_model': rnn_model,
        'rnn_forecast': rnn_forecast,
        'fit1_avg_total_time': fit_avg_total_time,
        'fit2_total_obs': fit_total_obs,
        'fit5_memory_peak': fit_memory_peak,
        'pred1_avg_series_per_sec': pred_avg_series_per_sec,
        'pred2_memory_peak': pred_memory_peak
        }