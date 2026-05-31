def create_rfm(df):

    # Simulated RFM values

    df["Recency"] = 100 - df["Spending Score (1-100)"]

    df["Frequency"] = df["Annual Income (k$)"] // 5

    df["Monetary"] = df["Annual Income (k$)"]

    return df