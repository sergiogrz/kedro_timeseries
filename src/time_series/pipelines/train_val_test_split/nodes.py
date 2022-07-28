from typing import Dict, Tuple
import pandas as pd
import logging

def split_data(data: pd.DataFrame, train_val_test_split: Dict) -> Dict:

    logger = logging.getLogger(__name__)

    val_size = train_val_test_split['val_size']
    test_size = train_val_test_split['test_size']

    train = data.iloc[ :-(val_size + test_size)]
    val = data.iloc[-(val_size + test_size): -test_size]
    test = data.iloc[-test_size:]

    logger.info(f'Períodos de entrenamiento: {len(train)}')
    logger.info(f'Períodos de validación: {len(val)}')
    logger.info(f'Períodos de test: {len(test)}')
    
    return {
        'train': train,
        'val': val,
        'train_val': pd.concat([train, val]),
        'test': test
    }