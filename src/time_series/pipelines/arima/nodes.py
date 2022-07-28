from typing import Dict
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ARIMA
import pickle
import logging
import time
import tracemalloc


def fit_arima(train: pd.DataFrame, fit_arima_p: Dict, complexity: Dict) -> Dict:

    logger = logging.getLogger(__name__)

    # Ejecución iterativa ('n_trials') para calcular tiempos y uso de memoria
    # El modelo que devuelve es el que se guarda en la última iteración
    total_times = []
    memory_peaks = []

    for trial in range(complexity['n_trials']):

        arima_models = {}

        start_time = time.process_time()  # time.process_time()  time.perf_counter()
        tracemalloc.start()

        for i in range(1, len(train.columns) + 1):
            logger.info(f'Ajuste {trial+1} de modelo para serie y{i}')
            arima_model = ARIMA(p = fit_arima_p['p'],
                                d = fit_arima_p['d'],
                                q = fit_arima_p['q'],
                                seasonal_order = list(map(int, fit_arima_p['seasonal_order'].split(','))),
                                trend = fit_arima_p['trend'],
                                random_state = 123)
            arima_model.fit(TimeSeries.from_dataframe(train.loc[:, [f'y{i}']]))
            arima_models[f'y{i}'] = arima_model
        
        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        total_times.append(end_time - start_time)
        memory_peaks.append(mem_peak / 10**6)

        for model in arima_models:
            arima_models[model] = pickle.dumps(arima_models[model])


    avg_total_time = np.mean(total_times)
    total_obs = train.shape[0] * train.shape[1]
    avg_series_time = np.mean(total_times) / train.shape[1]
    series_obs = train.shape[0]
    memory_peak = np.max(memory_peaks)

    return {
        'arima_models': arima_models,
        'fit1_avg_total_time': avg_total_time,
        'fit2_total_obs': total_obs,
        'fit3_avg_series_time': avg_series_time,
        'fit4_series_obs': series_obs,
        'fit5_memory_peak': memory_peak
        }


def predict_arima(arima_models: Dict, predict: Dict, complexity: Dict) -> Dict:

    logger = logging.getLogger(__name__)

    # Ejecución iterativa ('n_trials') para calcular tiempos y uso de memoria
    # La predicción que devuelve es la que se guarda en la última iteración
    n_series_per_sec = []
    memory_peaks = []

    for trial in range(complexity['n_trials']):

        logger.info(f'Predicción {trial+1}')

        arima_forecasts = {}

        start_time = time.process_time()  # time.process_time()  time.perf_counter()
        tracemalloc.start()

        for k in arima_models.keys():
            logger.info(f'Predicción {trial+1} de modelo para serie {k}')
            model = pickle.loads(arima_models[k])
            arima_forecasts[k] = model.predict(n = predict['n_periods'])
    
        arima_forecasts_list = [arima_forecasts[k].pd_dataframe() for k in arima_forecasts.keys()]
        arima_forecast = pd.concat(arima_forecasts_list, axis = 1)

        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        n_series_per_sec.append(len(arima_models) / (end_time - start_time))
        memory_peaks.append(mem_peak / 10**6)

    avg_series_per_sec = np.mean(n_series_per_sec)
    memory_peak = np.max(memory_peaks)

    return {
        'arima_forecast': arima_forecast,
        'pred1_avg_series_per_sec': avg_series_per_sec,
        'pred2_memory_peak': memory_peak
        }
