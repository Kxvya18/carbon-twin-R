import pandas as pd

def descriptive_statistics(df: pd.DataFrame):
    numeric = df.select_dtypes(include="number")
    desc = numeric.describe().T
    desc["skew"] = numeric.skew()
    desc["kurtosis"] = numeric.kurtosis()
    return desc
