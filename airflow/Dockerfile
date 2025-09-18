FROM apache/airflow:2.7.1-python3.12

ENV AIRFLOW_HOME=/app/airflow
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR $AIRFLOW_HOME

USER root

RUN apt-get update -y \
    && apt-get install -y python3-dev procps default-jre curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml poetry.lock* ./

USER airflow

RUN pip install --no-cache-dir poetry

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi --without dev

COPY airflow/ /app/airflow/

EXPOSE 8080
