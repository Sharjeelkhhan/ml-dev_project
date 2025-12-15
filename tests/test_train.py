import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import load_data, train_model

def test_data_loading():
    data = load_data("data/dataset.csv")
    assert isinstance(data, pd.DataFrame)

def test_model_training():
    data = load_data("data/dataset.csv")
    model, _ = train_model(data)
    assert model is not None

def test_shape_validation():
    data = load_data("data/dataset.csv")
    _, shape = train_model(data)
    assert shape[0] > 0 and shape[1] > 0
