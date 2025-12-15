import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

def load_data(path):
    return pd.read_csv(path)

def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, X.shape

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    data = load_data("data/dataset.csv")
    model, _ = train_model(data)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved successfully.")
