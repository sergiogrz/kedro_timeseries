"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline, pipeline
from kedro.pipeline.modular_pipeline import pipeline as mod_pipeline

from time_series.pipelines import (
    data_processing as dp, 
    train_val_test_split as tvt,
    baseline_naive as naive,
    arima as arima,
    autoarima as autoarima,
    prophet as prophet,
    nbeats as nbeats,
    rnn as rnn,
    model_evaluation as ev
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    data_processing_pipeline = dp.create_pipeline()
    train_val_test_pipeline = tvt.create_pipeline()
    naive_pipeline = naive.create_pipeline()
    arima_pipeline = arima.create_pipeline()
    autoarima_pipeline = autoarima.create_pipeline()
    prophet_pipeline = prophet.create_pipeline()
    nbeats_pipeline = nbeats.create_pipeline()
    rnn_pipeline = rnn.create_pipeline()
    model_evaluation_pipeline = ev.create_pipeline()

    return {
        "__default__": data_processing_pipeline
                    + train_val_test_pipeline,

        "data_processing": data_processing_pipeline,
        "train_val_test": train_val_test_pipeline,

        "naive_val": mod_pipeline(pipe = naive_pipeline, inputs = {'train': 'train'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'test': 'val', 'forecast': 'naive_forecast', 'train': 'train'}),
        "naive_test": mod_pipeline(pipe = naive_pipeline, inputs = {'train': 'train_val'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'train': 'train_val', 'test': 'test', 'forecast': 'naive_forecast'}),
        "naive_train_all": mod_pipeline(pipe = naive_pipeline, inputs = {'train': 'preprocessed_dataset'}),

        "arima_val": mod_pipeline(pipe = arima_pipeline, inputs = {'train': 'train'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'test': 'val', 'forecast': 'arima_forecast'}),
        "arima_test": mod_pipeline(pipe = arima_pipeline, inputs = {'train': 'train_val'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'train': 'train_val', 'test': 'test', 'forecast': 'arima_forecast'}),
        "arima_train_all": mod_pipeline(pipe = arima_pipeline, inputs = {'train': 'preprocessed_dataset'}),

        "autoarima_val": mod_pipeline(pipe = autoarima_pipeline, inputs = {'train': 'train'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'test': 'val', 'forecast': 'autoarima_forecast'}),
        "autoarima_test": mod_pipeline(pipe = autoarima_pipeline, inputs = {'train': 'train_val'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'train': 'train_val', 'test': 'test', 'forecast': 'autoarima_forecast'}),
        "autoarima_train_all": mod_pipeline(pipe = autoarima_pipeline, inputs = {'train': 'preprocessed_dataset'}),

        "prophet_val": mod_pipeline(pipe = prophet_pipeline, inputs = {'train': 'train'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'test': 'val', 'forecast': 'prophet_forecast'}),
        "prophet_test": mod_pipeline(pipe = prophet_pipeline, inputs = {'train': 'train_val'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'train': 'train_val', 'test': 'test', 'forecast': 'prophet_forecast'}),
        "prophet_train_all": mod_pipeline(pipe = prophet_pipeline, inputs = {'train': 'preprocessed_dataset'}),

        "nbeats_val": mod_pipeline(pipe = nbeats_pipeline, inputs = {'train': 'train'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'test': 'val', 'forecast': 'nbeats_forecast'}),
        "nbeats_test": mod_pipeline(pipe = nbeats_pipeline, inputs = {'train': 'train_val'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'train': 'train_val', 'test': 'test', 'forecast': 'nbeats_forecast'}),
        "nbeats_train_all": mod_pipeline(pipe = nbeats_pipeline, inputs = {'train': 'preprocessed_dataset'}),

        "rnn_val": mod_pipeline(pipe = rnn_pipeline, inputs = {'train': 'train'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'test': 'val', 'forecast': 'rnn_forecast'}),
        "rnn_test": mod_pipeline(pipe = rnn_pipeline, inputs = {'train': 'train_val'})
                    + mod_pipeline(pipe = model_evaluation_pipeline, inputs = {'train': 'train_val', 'test': 'test', 'forecast': 'rnn_forecast'}),
        "rnn_train_all": mod_pipeline(pipe = rnn_pipeline, inputs = {'train': 'preprocessed_dataset'}),
    }
