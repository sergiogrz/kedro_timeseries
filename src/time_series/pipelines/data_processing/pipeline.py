from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func = preprocess_dataset,
                inputs = 'companies',             # 'electric_production', 'companies', 'companies_red', 'companies_500'
                outputs = 'preprocessed_dataset',
                name = 'preprocess_dataset_node',
            )

        ]
    )
