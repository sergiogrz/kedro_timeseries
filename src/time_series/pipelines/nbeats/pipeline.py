from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fit_predict_nbeats # fit_nbeats, predict_nbeats

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = fit_predict_nbeats,
                inputs = ['train', 'params:fit_nbeats', 'params:complexity', 'params:predict'],
                outputs = {
                    'nbeats_model': 'nbeats_model',
                    'nbeats_forecast': 'nbeats_forecast',
                    'fit1_avg_total_time': 'fit1_avg_total_time',
                    'fit2_total_obs': 'fit2_total_obs',
                    'fit5_memory_peak': 'fit5_memory_peak',
                    'pred1_avg_series_per_sec': 'pred1_avg_series_per_sec',
                    'pred2_memory_peak': 'pred2_memory_peak'
                },
                name = 'fit_predict_nbeats_node',
            )
        ]
    )