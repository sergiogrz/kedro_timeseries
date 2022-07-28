import pandas as pd
from darts import TimeSeries

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:

    series = TimeSeries.from_dataframe(df, time_col='date')

    return series.pd_dataframe()