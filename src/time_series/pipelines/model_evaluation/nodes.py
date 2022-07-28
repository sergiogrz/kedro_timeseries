import logging
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

def _mape(y, y_hat):
    return np.mean(np.abs((y - y_hat)/y)*100)

def _mae(y, y_hat):
    return np.mean(np.abs(y - y_hat))


def evaluate_mape(test: pd.DataFrame, forecast: pd.DataFrame, evaluate: Dict) -> Dict:

    horizon = len(forecast)
    w = evaluate['window_forecast']
    
    # Medida global
    mape_values = [_mape(test[y], forecast[y]) for y in test.columns if not forecast[y].isnull().values.any()]
    # pd.Series([1,2, None]).isnull().values.any()
    
    names = ['mape_avg', 'mape_p25', 'mape_p50', 'mape_p75', 'mape_p95', 'mape_p99']
    values = np.concatenate((np.array([np.mean(mape_values)]), np.percentile(mape_values, [25, 50, 75, 95, 99])))
    mape_global = dict(zip(names, values))
    
    # Medida por ventanas
    mape_w = {}
    period = 1
    for i in range(0, horizon, w):
        mape_w_values = [_mape(test[i:i+w][y], forecast[i:i+w][y]) for y in test.columns if not forecast[y].isnull().values.any()]
        mape_w[f'mape_w{period}_avg'] = np.mean(mape_w_values)
        mape_w[f'mape_w{period}_p75'] = np.percentile(mape_w_values, 75)
        period += 1

    
    return {**mape_global, **mape_w}


def evaluate_mase(test: pd.DataFrame, forecast: pd.DataFrame, train: pd.DataFrame, evaluate: Dict) -> Dict:

    horizon = len(forecast)
    w = evaluate['window_forecast']

    ## Medida global
    # MAE
    mae_values = [_mae(test[y], forecast[y]) for y in test.columns if not forecast[y].isnull().values.any()]
    # MAE train naive
    naive_hat = train.iloc[:-1, :].reset_index().drop(columns = ['date'])
    naive = train.iloc[1:, :].reset_index().drop(columns = ['date'])
    mae_in_sample_values = [_mae(naive[y], naive_hat[y]) for y in test.columns if not forecast[y].isnull().values.any()]

    mase_values = [m / ms for m, ms in zip(mae_values, mae_in_sample_values)]

    names = ['mase_avg', 'mase_p25', 'mase_p50', 'mase_p75', 'mase_p95', 'mase_p99']
    values = np.concatenate((np.array([np.mean(mase_values)]), np.percentile(mase_values, [25, 50, 75, 95, 99])))
    mase_global = dict(zip(names, values))

    ## Medida por ventanas
    mase_w = {}
    period = 1
    for i in range(0, horizon, w):
        mae_w_values = [_mae(test[i:i+w][y], forecast[i:i+w][y]) for y in test.columns if not forecast[y].isnull().values.any()]
        mase_w_values = [m / ms for m, ms in zip(mae_w_values, mae_in_sample_values)]
        mase_w[f'mase_w{period}_avg'] = np.mean(mase_w_values)
        mase_w[f'mase_w{period}_p75'] = np.percentile(mase_w_values, 75)
        period += 1

    return {**mase_global, **mase_w}

