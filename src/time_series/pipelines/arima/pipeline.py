from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fit_arima, predict_arima

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = fit_arima,
                inputs = ['train', 'params:fit_arima', 'params:complexity'],
                outputs = {
                    'arima_models': 'arima_models',
                    'fit1_avg_total_time': 'fit1_avg_total_time',
                    'fit2_total_obs': 'fit2_total_obs',
                    'fit3_avg_series_time': 'fit3_avg_series_time',
                    'fit4_series_obs': 'fit4_series_obs',
                    'fit5_memory_peak': 'fit5_memory_peak'
                },
                name = 'fit_arima_node',
            ),
            node(
                func = predict_arima,
                inputs = ['arima_models', 'params:predict', 'params:complexity'],
                outputs = {
                    'arima_forecast': 'arima_forecast',
                    'pred1_avg_series_per_sec': 'pred1_avg_series_per_sec',
                    'pred2_memory_peak': 'pred2_memory_peak'
                },
                name = 'predict_arima_node',
            ),
        ]
    )