import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import KNNImputer

warnings.filterwarnings("ignore")

df = pd.read_csv("titanic_data.csv", skipinitialspace=True)

df.info()

df.drop(columns=['class', 'who', 'adult_male', 'alive'], inplace=True, axis=1)

print(df.info())