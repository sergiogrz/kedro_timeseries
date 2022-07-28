from typing import Dict
import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import AutoARIMA
import pickle
import logging
import time
import tracemalloc


def fit_autoarima(train: pd.DataFrame, fit_autoarima_p: Dict, complexity: Dict) -> Dict:

    logger = logging.getLogger(__name__)

    # Ejecución iterativa ('n_trials') para calcular tiempos y uso de memoria
    # El modelo que devuelve es el que se guarda en la última iteración
    total_times = []
    memory_peaks = []

    for trial in range(complexity['n_trials']):

        autoarima_models = {}

        start_time = time.process_time()  # time.process_time()  time.perf_counter()
        tracemalloc.start()

        for i in range(1, len(train.columns) + 1):
            try:
                logger.info(f'Ajuste {trial+1} de modelo para serie y{i}')
                autoarima_model = AutoARIMA(start_p = fit_autoarima_p['start_p'],
                                            start_q = fit_autoarima_p['start_q'],
                                            max_p = fit_autoarima_p['max_p'],
                                            max_q = fit_autoarima_p['max_q'],
                                            max_order = fit_autoarima_p['max_order'],
                                            m = fit_autoarima_p['m'],
                                            random_state = 123)
                autoarima_model.fit(TimeSeries.from_dataframe(train.loc[:, [f'y{i}']]))
                autoarima_models[f'y{i}'] = autoarima_model

                logger.info(f'Values of (p, d, q): {autoarima_model.model.model_.order}')
                logger.info(f'Values of (P, D, Q, m): {autoarima_model.model.model_.seasonal_order}')
            
            except:
                autoarima_models[f'y{i}'] = None
    
        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        total_times.append(end_time - start_time)
        memory_peaks.append(mem_peak / 10**6)

        for model in autoarima_models:
            autoarima_models[model] = pickle.dumps(autoarima_models[model])

    avg_total_time = np.mean(total_times)
    total_obs = train.shape[0] * train.shape[1]
    avg_series_time = np.mean(total_times) / train.shape[1]
    series_obs = train.shape[0]
    memory_peak = np.max(memory_peaks)

    return {
        'autoarima_models': autoarima_models,
        'fit1_avg_total_time': avg_total_time,
        'fit2_total_obs': total_obs,
        'fit3_avg_series_time': avg_series_time,
        'fit4_series_obs': series_obs,
        'fit5_memory_peak': memory_peak
        }


def predict_autoarima(autoarima_models: Dict, predict: Dict, complexity: Dict) -> Dict:

    logger = logging.getLogger(__name__)

    # Ejecución iterativa ('n_trials') para calcular tiempos y uso de memoria
    # La predicción que devuelve es la que se guarda en la última iteración
    n_series_per_sec = []
    memory_peaks = []

    for trial in range(complexity['n_trials']):

        logger.info(f'Predicción {trial+1}')

        autoarima_forecasts = {}

        start_time = time.process_time()  # time.process_time()  time.perf_counter()
        tracemalloc.start()

        for k in autoarima_models.keys():
            logger.info(f'Predicción {trial+1} de modelo para serie {k}')
            model = pickle.loads(autoarima_models[k])
            if model is None:
                autoarima_forecasts[k] = TimeSeries.from_series(pd.Series([None for i in range(predict['n_periods'])]))
            else:
                autoarima_forecasts[k] = model.predict(n = predict['n_periods'])
        
        autoarima_forecasts_list = [autoarima_forecasts[k].pd_dataframe() for k in autoarima_forecasts.keys()]
        autoarima_forecast = pd.concat(autoarima_forecasts_list, axis = 1)

        end_time = time.process_time()   # time.process_time()  time.perf_counter()
        mem_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()

        n_series_per_sec.append(len(autoarima_models) / (end_time - start_time))
        memory_peaks.append(mem_peak / 10**6)

    avg_series_per_sec = np.mean(n_series_per_sec)
    memory_peak = np.max(memory_peaks)

    return {
        'autoarima_forecast': autoarima_forecast,
        'pred1_avg_series_per_sec': avg_series_per_sec,
        'pred2_memory_peak': memory_peak
        }