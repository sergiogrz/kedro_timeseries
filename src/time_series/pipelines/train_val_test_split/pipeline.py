from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = split_data,
                inputs = ['preprocessed_dataset', 'params:train_val_test_split'],
                outputs = {
                    'train': 'train',
                    'val': 'val',
                    'train_val': 'train_val',
                    'test': 'test'
                },
                name = 'split_data_node',
            )
        ]
    )