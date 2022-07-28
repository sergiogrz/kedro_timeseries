from kedro.pipeline import Pipeline, node, pipeline
from .nodes import evaluate_mape, evaluate_mase


def create_pipeline(**kwargs) -> Pipeline:

    return pipeline(
        [
            node(
                func = evaluate_mape,
                inputs = ['test',
                        'forecast',
                        'params:evaluate'],
                outputs = {
                    'mape_avg': 'MAPE_avg',
                    'mape_p25': 'MAPE_p25',
                    'mape_p50': 'MAPE_p50',
                    'mape_p75': 'MAPE_p75',
                    'mape_p95': 'MAPE_p95',
                    'mape_p99': 'MAPE_p99',
                    'mape_w1_avg': 'MAPE_w1_avg',
                    'mape_w1_p75': 'MAPE_w1_p75',
                    'mape_w2_avg': 'MAPE_w2_avg',
                    'mape_w2_p75': 'MAPE_w2_p75',
                    'mape_w3_avg': 'MAPE_w3_avg',
                    'mape_w3_p75': 'MAPE_w3_p75',
                    'mape_w4_avg': 'MAPE_w4_avg',
                    'mape_w4_p75': 'MAPE_w4_p75',
                },
                name = 'evaluate_mape_node'
            ),
            node(
                func = evaluate_mase,
                inputs = ['test',
                        'forecast',
                        'train',
                        'params:evaluate'],
                outputs = {
                    'mase_avg': 'MASE_avg',
                    'mase_p25': 'MASE_p25',
                    'mase_p50': 'MASE_p50',
                    'mase_p75': 'MASE_p75',
                    'mase_p95': 'MASE_p95',
                    'mase_p99': 'MASE_p99',
                    'mase_w1_avg': 'MASE_w1_avg',
                    'mase_w1_p75': 'MASE_w1_p75',
                    'mase_w2_avg': 'MASE_w2_avg',
                    'mase_w2_p75': 'MASE_w2_p75',
                    'mase_w3_avg': 'MASE_w3_avg',
                    'mase_w3_p75': 'MASE_w3_p75',
                    'mase_w4_avg': 'MASE_w4_avg',
                    'mase_w4_p75': 'MASE_w4_p75',
                },
                name = 'evaluate_mase_node'
            ),
        ]
    )