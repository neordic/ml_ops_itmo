from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def greet():
    print("Привет от Airflow")


default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
}

with DAG(
    dag_id="hello_dag",
    default_args=default_args,
    schedule_interval="@once",
    catchup=False,
) as dag:
    greet_task = PythonOperator(task_id="greet_task", python_callable=greet)
