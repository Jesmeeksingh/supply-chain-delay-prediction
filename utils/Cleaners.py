import pandas as pd
def clean_weight_column(df):
    """
    Cleans Weight (Kilograms) column:
    - Forces numeric
    - Removes invalid strings
    - Drops or imputes bad values
    """
    df = df.copy()

    df["Weight (Kilograms)"] = pd.to_numeric(
        df["Weight (Kilograms)"],
        errors="coerce"
    )

    # Remove zero or negative weights
    df = df[df["Weight (Kilograms)"] > 0]

    return df

