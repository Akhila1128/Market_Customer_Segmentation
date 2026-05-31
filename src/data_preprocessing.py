import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    df = pd.read_csv("dataset/Mall_Customers.csv")

    encoder = LabelEncoder()

    df["Gender"] = encoder.fit_transform(df["Gender"])

    return df