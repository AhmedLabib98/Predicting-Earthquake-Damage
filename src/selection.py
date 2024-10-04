import pandas as pd

def select_features(
        df: pd.DataFrame,
        columns_to_keep: list[str] = [],
        columns_to_drop: list[str] = [],
    ) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): DataFrame to select features from.
        columns_to_keep (list[str], optional): Columns to keep. Defaults to [].
        columns_to_drop (list[str], optional): Columns to drop. Defaults to [].

    Returns:
        pd.DataFrame: DataFrame with selected features.
    """
    if len(columns_to_drop) > 0:
        df = df.drop(columns=columns_to_drop)

    if len(columns_to_keep) > 0:
        df = df[columns_to_keep]

    return df