# Auto-Prompting Framework

**Auto-Prompting Framework** — универсальный инструмент для автоматизированной генерации, оптимизации и оценки промптов для больших языковых моделей (LLM).

## О проекте

В рамках данного проекта реализован фреймворк для автоматической разработки промптов с целью решения задач классификации и генерации текста. Основные возможности:

* **Методы оптимизации промптов**:

  * PromptV1 и PromptV2 (жёсткие токены-промпты).
  * Mixture Soft Prompting (смешанные мягкие токены).
  * TextGrad (градиентная оптимизация).
  * Consprompt (контрастивная оптимизация).
  * Prewrite RL (обучение с подкреплением).
  * APE (автоматический эволютивный поиск).
* **Поддержка экспериментов** через Hydra + YAML-конфигурации.
* **Интеграция** с `transformers`, `datasets`, `evaluate`, `openai`.
* **Юнит-тесты** на базе `pytest` для каждой реализации метода.

## Примеры использования кастомного метода

Релизация нашего кастомного метода для falcon и roberta-basae для ag_news, sst2, trec находится в ноутбуках:

* `src/methods/ptcprl_method_roberta.ipynb`
* `src/methods/ptcprl_method_falcon.ipynb`

## Зависимости

```bash
> pytest
> torch
> transformers
> datasets
> evaluate
> hydra-core
> omegaconf
> tqdm
> openai
```

## Структура проекта

```
.
├── configs/            # YAML-файлы конфигурации для экспериментов
├── src/                # Исходный код фреймворка
│   ├── run_experiment.py   # Точка входа Hydra
│   ├── data_utils.py       # Загрузка и подготовка данных
│   ├── callbacks/          # Трекер метрик по эпохам
│   └── methods/            # Реализация методов оптимизации промптов
├── utils/              # Вспомогательные функции
├── tests/              # Тесты (pytest)
├── run_all_experiments.ps1  # Скрипт для запуска всех конфигураций (Windows)
├── pytest.ini          # Конфигурация pytest
└── README.md           # Этот файл
```

## Использование

### Конфигурация

Все файлы `.yaml` в папке `configs` описывают связку:

```
<dataset>_<метод>_<модель>.yaml
```

Например: `ag_news_prompt_v1_roberta_base.yaml`.

### Запуск

```bash
python src/run_experiment.py --config-name ag_news_prompt_v1_roberta_base
```

Для частичного переопределения параметров:

```bash
python src/run_experiment.py --config-name ag_news_prompt_v1_roberta_base method_cfg.num_candidates=10 training.max_epochs=3
```

### Результаты

* Логи обучения сохраняются в `outputs/exp_name/YYYY-MM-DD_HH-MM-SS/`.
* Метрики по эпохам — в файле `epoch_metrics.jsonl`.

## Тестирование

```bash
pytest
```

## Добавление нового метода

1. Создать новый класс Runner в `src/methods` по образцу существующих.
2. Добавить запись в словарь `_METHODS` в `src/run_experiment.py`.
3. Написать YAML-конфиг в `configs`.
4. Добавить тесты в `tests/`.
