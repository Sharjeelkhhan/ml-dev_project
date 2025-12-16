from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("/opt/airflow/dags/data/dataset.csv")
    print("Data loaded")
    return df.shape

def train_model():
    df = pd.read_csv("/opt/airflow/dags/data/dataset.csv")
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    os.makedirs("/opt/airflow/dags/models", exist_ok=True)
    with open("/opt/airflow/dags/models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved")

with DAG(
    dag_id="training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    load_task = PythonOperator(
        task_id="load_data",
        python_callable=load_data
    )

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    load_task >> train_task
