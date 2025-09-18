from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


PREP_SCRIPT = "/app/airflow/dags/scripts/prep_data.py"
TRAIN_SCRIPT = "/app/airflow/dags/scripts/train.py"
TEST_SCRIPT = "/app/airflow/dags/scripts/testing.py"

# Файлы/папки, которые версионируем DVC на шагах
PROCESSED = "data/processed/train_simple.csv data/processed/test_simple.csv data/processed/train_fe.csv data/processed/test_fe.csv"
MODELS = "models/logreg_simple.pkl models/rf_simple.pkl models/mlp_simple.pkl models/logreg_fe.pkl models/rf_fe.pkl models/mlp_fe.pkl"
METRICS = "results/metrics.csv"

default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 1, 1),
}

# Общий helper-шелл для безопасной авторизации DVC (если креды заданы)
DVC_AUTH_SNIPPET = r"""
cd /app/airflow/
set -e
if [ -n "${DAGSHUB_USER:-}" ] && [ -n "${DAGSHUB_TOKEN:-}" ]; then
  dvc remote modify --local origin-dags auth basic || true
  dvc remote modify --local origin-dags user "$DAGSHUB_USER" || true
  dvc remote modify --local origin-dags password "$DAGSHUB_TOKEN" || true
else
  echo "DVC auth: DAGSHUB_USER/TOKEN not provided — pushing may be skipped."
fi
"""

with DAG(
    dag_id="titanic_hw3",
    default_args=default_args,
    schedule_interval=None,  # запуск вручную из UI
    catchup=False,
    tags=["titanic", "hw3", "dvc"],
) as dag:
    # 0) Мягкий dvc pull (если offline — не падаем)
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        bash_command=DVC_AUTH_SNIPPET
        + r"""
          cd /app/airflow/
          set -e
          dvc pull || echo "dvc pull failed or no remote — continue offline"
        """,
        # Пробрасываем креды через переменные окружения (заведи их в .env/compose или Airflow Variables)
        env={
            # эти значения можно задать в .env → docker compose, или в UI: Admin → Variables
            "DAGSHUB_USER": "{{ var.value.get('DAGSHUB_USER', '') }}",
            "DAGSHUB_TOKEN": "{{ var.value.get('DAGSHUB_TOKEN', '') }}",
        },
    )

    # 1) Препроцессинг
    prep_data = BashOperator(
        task_id="prep_data",
        bash_command=DVC_AUTH_SNIPPET
        + rf"""
          cd /app/airflow/
          set -e
          python {PREP_SCRIPT}
          dvc add {PROCESSED} || true
          dvc push || echo "No remote or auth — skip push"
        """,
        env={
            "DAGSHUB_USER": "{{ var.value.get('DAGSHUB_USER', '') }}",
            "DAGSHUB_TOKEN": "{{ var.value.get('DAGSHUB_TOKEN', '') }}",
        },
    )

    # 2) Обучение моделей
    train_models = BashOperator(
        task_id="train_models",
        bash_command=DVC_AUTH_SNIPPET
        + rf"""
          cd /app/airflow/
          set -e
          python {TRAIN_SCRIPT}
          dvc add {MODELS} || true
          dvc push || echo "No remote or auth — skip push"
        """,
        env={
            "DAGSHUB_USER": "{{ var.value.get('DAGSHUB_USER', '') }}",
            "DAGSHUB_TOKEN": "{{ var.value.get('DAGSHUB_TOKEN', '') }}",
        },
    )

    # 3) Тестирование/метрики
    evaluate = BashOperator(
        task_id="evaluate",
        bash_command=DVC_AUTH_SNIPPET
        + rf"""
          cd /app/airflow/
          set -e
          python {TEST_SCRIPT}
          if [ -f {METRICS} ]; then
            dvc add {METRICS} || true
            dvc push || echo "No remote or auth — skip push"
          else
            echo "metrics.csv not found"
          fi
        """,
        env={
            "DAGSHUB_USER": "{{ var.value.get('DAGSHUB_USER', '') }}",
            "DAGSHUB_TOKEN": "{{ var.value.get('DAGSHUB_TOKEN', '') }}",
        },
    )

    dvc_pull >> prep_data >> train_models >> evaluate
