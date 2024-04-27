import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder


def onehot_columns(df, columns):
    OHE = OneHotEncoder(sparse_output=False)
    OHE.fit(df[columns])
    transformed = OHE.transform(df[columns])
    transformed_df = pd.DataFrame(transformed, columns=OHE.get_feature_names_out())

    return transformed_df

def merge_onehotted(df, transformed_df, columns):
    
    return pd.concat([df, transformed_df], axis=1).drop(columns, axis=1)