# This file will load the data and we will apply the memory reduction top 

import numpy as np
import pandas as pd
from pathlib import Path
from src.data_processing.data_size_reduction import reduce_memory_usage
from src.config import (
    TRAIN_TRANSACTIONS, TRAIN_IDENTITY, TEST_TRANSACTIONS, TEST_IDENTITY
)

def data_loading(transaction_data: Path | str, identity_data: Path | str) -> pd.DataFrame:
    
    """
        This function loads the datasets and then apply the memory reduction on them individually and then apply a left join on them.
    """
    transaction_df= pd.read_csv(transaction_data)
    print(transaction_df.shape)
    transaction_df_reduced= reduce_memory_usage(df= transaction_df)
    
    identity_df= pd.read_csv(identity_data)
    print(identity_df.shape)
    identity_df_reduced= reduce_memory_usage(df= identity_df)
    
    # merge them both
    overall_df= pd.merge(transaction_df_reduced, identity_df_reduced, how="left")
    print(f"Dataset shape: {overall_df.shape}")
    
    return overall_df


if __name__ == "__main__":
    
    train_data= data_loading(
        transaction_data= TRAIN_TRANSACTIONS, identity_data= TRAIN_IDENTITY
    )
    print(train_data)
    test_data= data_loading(
        transaction_data= TEST_TRANSACTIONS, identity_data= TEST_IDENTITY
    )
    print(test_data)
    
