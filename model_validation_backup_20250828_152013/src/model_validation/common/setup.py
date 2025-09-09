import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42

def get_data_path(filename="sample_data.csv"):
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    return os.path.join(ROOT, "data", filename)
