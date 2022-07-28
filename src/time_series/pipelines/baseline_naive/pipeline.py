from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fit_naive, predict_naive

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = fit_naive,
                inputs = ['train', 'params:fit_naive', 'params:complexity'],
                outputs = {
                    'naive_models': 'naive_models',
                    'fit1_avg_total_time': 'fit1_avg_total_time',
                    'fit2_total_obs': 'fit2_total_obs',
                    'fit3_avg_series_time': 'fit3_avg_series_time',
                    'fit4_series_obs': 'fit4_series_obs',
                    'fit5_memory_peak': 'fit5_memory_peak'
                },
                name = 'fit_naive_node',
            ),
            node(
                func = predict_naive,
                inputs = ['naive_models', 'params:predict', 'params:complexity'],
                outputs = {
                    'naive_forecast': 'naive_forecast',
                    'pred1_avg_series_per_sec': 'pred1_avg_series_per_sec',
                    'pred2_memory_peak': 'pred2_memory_peak'
                },
                name = 'predict_naive_node',
            ),
        ]
    )