FROM python:3.12

WORKDIR /app

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

COPY . .

RUN poetry install --with dev

CMD ["poetry", "run", "python", "titanic/train.py"]
