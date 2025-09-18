# hw_1

- создан репо проекта
- за основу взят github flow (main и feature ветки)
- настроена работа линтеров/форматеров с помощью `ruff`
- настроено управление зависимости с помощью `poetry`
- написан docker file для сборки окружения
- описан CI пайп

Чтобы собрать и запустить докер:

`docker build -t ml_ops_itmo .`


`docker run --rm ml_ops_itmo`

# hw_2

- выбрана система версионирования(dvc)
- добавлены сырые данные в систему версионирвоания (отдельным комитом)
- написаны скрипты для **обработки данных** (несколько датасетов) и добавлен результат работы в систему версионирвоания (отдельнымм комитами)

Данные(трейн\тест) без обработки(симпл):

`poetry run python scripts/prep_data.py`

`poetry run dvc add data/processed/train_simple.csv`

`poetry run dvc add data/processed/test_simple.csv`

`git add data/processed/train_simple.csv.dvc data/processed/test_simple.csv.dvc .gitignore`

`git commit -m "Version SIMPLE datasets with DVC"`

Данные(трейн\тест) с обработкой(feat eng):

`poetry run dvc add data/processed/train_fe.csv`

`poetry run dvc add data/processed/test_fe.csv`

`git add data/processed/train_fe.csv.dvc data/processed/test_fe.csv.dvc`

`git commit -m "Version FE datasets with DVC"`

Использует удаленное хранилище DagsHub

`poetry run dvc remote add origin-dags https://dagshub.com/nmuraveva08/ml_ops_itmo.dvc`

`poetry run dvc remote default origin-dags`

`poetry run dvc remote modify --local origin-dags auth basic`

`poetry run dvc remote modify --local origin-dags user nmuraveva08`

`poetry run dvc remote modify --local origin-dags password <МОЙ_ТОКЕН>`

`git add .dvc/config`
`git commit -m "Configure DVC remote (DagsHub)"`

`poetry run dvc push`

- написаны скрипты для **обучения разных моделей** на разных датасетах и обученные модели добавлены в систему версионирования (отдельными комитами)

simple models:

`poetry run dvc add models/logreg_simple.pkl `

`models/rf_simple.pkl models/mlp_simple.pkl`

`git add models/logreg_simple.pkl.dvc models/rf_simple.pkl.dvc models/mlp_simple.pkl.dvc .gitignore`

`git commit -m "Version models (SIMPLE): logreg, RF, MLP"`

FE-модели

`poetry run dvc add models/logreg_fe.pkl models/rf_fe.pkl models/mlp_fe.pkl`

`git add models/logreg_fe.pkl.dvc models/rf_fe.pkl.dvc models/mlp_fe.pkl.dvc`

`git commit -m "Version models (FE): logreg, RF, MLP"`

`poetry run dvc push`

- модели выгружены из системы версионирования. Результаты на тестовых выборках:

`poetry run python scripts/testing.py`

`poetry run dvc add results/metrics.csv`

`git add results/metrics.csv.dvc`

`git commit -m "Add evaluation metrics (models compared on test/holdout)"`

`poetry run dvc push`

Результаты

| Dataset | Model  | Accuracy |     F1 |
| ------: | :----- | -------: | -----: |
|      fe | rf     |   0.9888 | 0.9855 |
|      fe | logreg |   0.8268 | 0.7559 |
|      fe | mlp    |   0.8212 | 0.7714 |
|  simple | rf     |   0.9888 | 0.9855 |
|  simple | mlp    |   0.8101 | 0.7424 |
|  simple | logreg |   0.8101 | 0.7344 |

Вывод: лучшая модель - `rf на датасете fe` (accuracy = 0.9888, F1 = 0.9855).

# hw_3

- выбрать систему управления пайплайнами airflow. развернуть локально:
`docker compose up -d --build `
- запуск системы реализован с использованием docker
- написать пайплайн из трех шагов: обработка данных(`prep_data`), обучение модели(`train_models`), тестирование модели (`evaluate`)
- пайплайн интегрирован с dvc
