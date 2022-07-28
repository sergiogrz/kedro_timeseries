from kedro.pipeline import Pipeline, node, pipeline
from .nodes import fit_predict_rnn

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = fit_predict_rnn,
                inputs = ['train', 'params:fit_rnn', 'params:complexity', 'params:predict'],
                outputs = {
                    'rnn_model': 'rnn_model',
                    'rnn_forecast': 'rnn_forecast',
                    'fit1_avg_total_time': 'fit1_avg_total_time',
                    'fit2_total_obs': 'fit2_total_obs',
                    'fit5_memory_peak': 'fit5_memory_peak',
                    'pred1_avg_series_per_sec': 'pred1_avg_series_per_sec',
                    'pred2_memory_peak': 'pred2_memory_peak'
                },
                name = 'fit_predict_rnn_node',
            )
        ]
    )