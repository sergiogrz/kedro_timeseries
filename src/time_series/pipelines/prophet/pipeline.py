from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fit_prophet, predict_prophet

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = fit_prophet,
                inputs = ['train', 'params:fit_prophet', 'params:complexity'],
                outputs = {
                    'prophet_models': 'prophet_models',
                    'fit1_avg_total_time': 'fit1_avg_total_time',
                    'fit2_total_obs': 'fit2_total_obs',
                    'fit3_avg_series_time': 'fit3_avg_series_time',
                    'fit4_series_obs': 'fit4_series_obs',
                    'fit5_memory_peak': 'fit5_memory_peak'
                },
                name = 'fit_prophet_node',
            ),
            node(
                func = predict_prophet,
                inputs = ['prophet_models', 'params:predict', 'params:complexity'],
                outputs = {
                    'prophet_forecast': 'prophet_forecast',
                    'pred1_avg_series_per_sec': 'pred1_avg_series_per_sec',
                    'pred2_memory_peak': 'pred2_memory_peak'
                },
                name = 'predict_prophet_node',
            ),
        ]
    )