# Packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as nsn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

# Data loading

df1 = pd.read_csv("C:/Users/chole/Desktop/UCI_Heart_Disease/df_1.csv")
df2 = pd.read_csv("C:/Users/chole/Desktop/UCI_Heart_Disease/df_2.csv")

# Data split

X1 = df1.drop(columns="num")
y1 = df1.num

X2 = df2.drop(columns="num")
y2 = df2.num

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1,
                                                        test_size=0.2,
                                                        random_state=42)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2,
                                                        test_size=0.2,
                                                        random_state=42)

# Column transformer for df1 and df2

column_t_1 = make_column_transformer(
    (OneHotEncoder(drop="if_binary"), ["sex", "cp", "dataset", "fbs", "restecg", "exang", "slope"]),
    (StandardScaler(), ["age", "trestbps", "chol", "thalch", "oldpeak"]),
    remainder="passthrough"
)

column_t_2 = make_column_transformer(
    (OneHotEncoder(drop="if_binary"), ["sex", "cp", "dataset", "fbs", "restecg", "exang"]),
    (StandardScaler(), ["age", "trestbps", "chol", "thalch", "oldpeak"]),
    remainder="passthrough"
)

