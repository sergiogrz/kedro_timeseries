from typing import Dict
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import NaiveSeasonal
import pickle
import logging
import time
import tracemalloc


def fit_naive(train: pd.DataFrame, fit_naive_p: Dict, complexity: Dict) -> Dict:

    logger = logging.getLogger(__name__)

    # Ejecución iterativa ('n_trials') para calcular tiempos y uso de memoria
    # El modelo que devuelve es el que se guarda en la última iteración
    total_times = []
    memory_peaks = []

    for trial in range(complexity['n_trials']):

        naive_models = {}

        start_time = time.process_time()  # time.process_time()  time.perf_counter()
        tracemalloc.start()

        for i in range(1, len(train.columns) + 1):
            logger.info(f'Ajuste {trial+1} de modelo para serie y{i}')
            naive_model = NaiveSeasonal(K=1)
            naive_model.fit(TimeSeries.from_dataframe(train.loc[:, [f'y{i}']]))
            naive_models[f'y{i}'] = naive_model

        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        total_times.append(end_time - start_time)
        memory_peaks.append(mem_peak / 10**6)

        for model in naive_models:
            naive_models[model] = pickle.dumps(naive_models[model])
    
    avg_total_time = np.mean(total_times)
    total_obs = train.shape[0] * train.shape[1]
    avg_series_time = np.mean(total_times) / train.shape[1]
    series_obs = train.shape[0]
    memory_peak = np.max(memory_peaks)
    
    return {
        'naive_models': naive_models,
        'fit1_avg_total_time': avg_total_time,
        'fit2_total_obs': total_obs,
        'fit3_avg_series_time': avg_series_time,
        'fit4_series_obs': series_obs,
        'fit5_memory_peak': memory_peak
        }


def predict_naive(naive_models: Dict, predict: Dict, complexity: Dict) -> Dict:

    logger = logging.getLogger(__name__)

    # Ejecución iterativa ('n_trials') para calcular tiempos y uso de memoria
    # La predicción que devuelve es la que se guarda en la última iteración
    n_series_per_sec = []
    memory_peaks = []

    for trial in range(complexity['n_trials']):

        logger.info(f'Predicción {trial+1}')

        naive_forecasts = {}

        start_time = time.process_time()  # time.process_time()  time.perf_counter()
        tracemalloc.start()

        for k in naive_models.keys():
            model = pickle.loads(naive_models[k])
            naive_forecasts[k] = model.predict(n = predict['n_periods'])

        naive_forecasts_list = [naive_forecasts[k].pd_dataframe() for k in naive_forecasts.keys()]
        naive_forecast = pd.concat(naive_forecasts_list, axis = 1)

        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        n_series_per_sec.append(len(naive_models) / (end_time - start_time))
        memory_peaks.append(mem_peak / 10**6)

    avg_series_per_sec = np.mean(n_series_per_sec)
    memory_peak = np.max(memory_peaks)

    return {
        'naive_forecast': naive_forecast,
        'pred1_avg_series_per_sec': avg_series_per_sec,
        'pred2_memory_peak': memory_peak
        }