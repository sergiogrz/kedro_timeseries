from typing import Dict, Tuple
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
import pickle
import logging
import time
import tracemalloc


def fit_predict_nbeats(train: pd.DataFrame, fit_nbeats_p: Dict, complexity: Dict, predict: Dict) -> Dict:

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
        nbeats_model = NBEATSModel(input_chunk_length = fit_nbeats_p['input_chunk_length'],
                                    output_chunk_length = fit_nbeats_p['output_chunk_length'],
                                    generic_architecture = True,
                                    num_stacks = fit_nbeats_p['num_stacks'],
                                    num_blocks = fit_nbeats_p['num_blocks'],
                                    num_layers = fit_nbeats_p['num_layers'],
                                    layer_widths = fit_nbeats_p['layer_widths'],
                                    n_epochs = fit_nbeats_p['n_epochs'],
                                    random_state = 123)
        nbeats_model.fit(train_all_diff_scaled)

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
   
        ## Predicción
        nbeats_forecast_diff_scaled = nbeats_model.predict(n = predict['n_periods'], series = train_all_diff_scaled)

        # Desescalado
        nbeats_forecast_diff = scaler.inverse_transform(nbeats_forecast_diff_scaled)


        nbeats_forecast_diff_list = [nbeats_forecast_diff[i].pd_dataframe() for i in range(len(nbeats_forecast_diff))]
        nbeats_forecast_diff = pd.concat(nbeats_forecast_diff_list, axis = 1)

        # Integración = inversa diferenciación
        nbeats_forecast = nbeats_forecast_diff.copy()
        nbeats_forecast.iloc[0, :] = nbeats_forecast.iloc[0, :] + train.iloc[-1, :]
        nbeats_forecast = nbeats_forecast.cumsum(axis = 0)

        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        pred_n_series_per_sec.append(len(train.columns) / (end_time - start_time))
        pred_memory_peaks.append(mem_peak / 10**6)
    
    pred_avg_series_per_sec = np.mean(pred_n_series_per_sec)
    pred_memory_peak = np.max(pred_memory_peaks)

    nbeats_model = {'nbeats_model': pickle.dumps(nbeats_model)}

    return {
        'nbeats_model': nbeats_model,
        'nbeats_forecast': nbeats_forecast,
        'fit1_avg_total_time': fit_avg_total_time,
        'fit2_total_obs': fit_total_obs,
        'fit5_memory_peak': fit_memory_peak,
        'pred1_avg_series_per_sec': pred_avg_series_per_sec,
        'pred2_memory_peak': pred_memory_peak
        }

