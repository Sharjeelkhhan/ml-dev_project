import sys
import os
import pandas as pd
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import load_data, train_model


def create_dummy_csv():
    df = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [5, 6, 7, 8],
        "target": [0, 1, 0, 1]
    })

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    df.to_csv(temp_file.name, index=False)
    return temp_file.name


def test_data_loading():
    csv_path = create_dummy_csv()
    data = load_data(csv_path)
    assert isinstance(data, pd.DataFrame)


def test_model_training():
    csv_path = create_dummy_csv()
    data = load_data(csv_path)
    model = train_model(data)
    assert model is not None


def test_shape_validation():
    csv_path = create_dummy_csv()
    data = load_data(csv_path)
    assert data.shape[0] > 0 and data.shape[1] > 1
