# This file will load the data and we will apply the memory reduction top 

import numpy as np
import pandas as pd
from pathlib import Path
from src.data_processing.data_size_reduction import reduce_memory_usage
from src.data_processing.feature_engineering import Devices

def data_loading(transaction_data: Path | str, identity_data: Path | str) -> pd.DataFrame:
    
    """
        This function loads the datasets and then apply the memory reduction on them individually and then apply a left join on them.
    """
    transaction_df= pd.read_csv(transaction_data)
    print(transaction_df.shape)
    transaction_df_reduced= reduce_memory_usage(df= transaction_df)
    
    identity_df= pd.read_csv(identity_data)
    print(identity_df.shape)
    # test_identity uses "id-01" style names; make them match train's "id_01"
    identity_df.columns= identity_df.columns.str.replace('-', '_')
    # feature engineering runs before the merge so `had_id` marks matched rows 
    identity_df= Devices(identity_df)
    identity_df_reduced= reduce_memory_usage(df= identity_df)
    
    # merge them both
    overall_df= pd.merge(transaction_df_reduced, identity_df_reduced, how="left")
    # rows with no identity match get NaN for had_id; make it 0 so it's a real 0/1 flag
    overall_df['had_id']= overall_df['had_id'].fillna(0).astype(np.int8)
    print(f"Dataset shape: {overall_df.shape}")
    
    return overall_df

