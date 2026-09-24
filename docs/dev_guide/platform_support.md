# Установка и совместимость

Industrial использует один действующий контракт интеграции с FEDOT. Пакет
FEDOT закреплён на полном идентификаторе коммита
`d1875e7a1ba49c94d13c51d97ec78829bf459ca8` из влитого Pull Request FEDOT
[#1462](https://github.com/aimclub/FEDOT/pull/1462), который добавил новый
контракт расширений и `TensorData`. Профили `fedot-legacy` и `fedot-tensor`
удалены: выбирать реализацию через дополнительную группу или переменную
окружения больше не требуется.

## Источники настроек

| Файл | Назначение | Обновление |
| --- | --- | --- |
| `pyproject.toml` | Версия Python, основные зависимости и дополнительные группы | Вручную, вместе с тестами контракта |
| `requirements.txt` | Точный текстовый экспорт основных зависимостей | `python -m tools.platform_support export` |
| `uv.lock` | Разрешённое дерево зависимостей и источники пакетов | `uv lock`, затем `uv lock --check` |
| `tools/platform_support/compatibility.json` | Поддерживаемые версии Python, FEDOT SHA и известные ограничения | Вместе с доказательствами совместимости |

FEDOT должен быть указан ровно один раз среди основных зависимостей как прямая
Git-ссылка на полный SHA. В дополнительных группах повторно объявлять FEDOT
запрещено. Команда `check` проверяет это правило, диапазон Python, обязательные
ограничения и соответствие `requirements.txt` данным из `pyproject.toml`.

## Поддерживаемые версии

| Python | Статус | Проверки |
| --- | --- | --- |
| 3.10 | Поддерживается | Полные модульные тесты, интеграция с FEDOT, сборка и проверка wheel |
| 3.11 | Поддерживается | Интеграция с FEDOT, установка и проверка wheel |
| 3.12 | Только аудит | Метаданные wheel должны отклонять установку |
| 3.9 и ниже | Не поддерживается | Установка отклоняется метаданными |

Версия пакета FEDOT не доказывает происхождение исходного кода. Команда
`environment` сверяет запись PEP 610 из `direct_url.json` с репозиторием и SHA,
указанными в политике совместимости.

## Установка разработчика

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "uv==0.8.17"
uv sync --extra dev
python -m pip check
```

Для CPU-среды в CI PyTorch устанавливается отдельно из официального индекса,
после чего остальные пакеты устанавливаются из `uv.lock`. Наличие CUDA не
следует выводить из успешной установки: вычисления на CUDA проверяются
отдельными сценариями.

## Интеграция с FEDOT

Операции Industrial описаны в
`fedot_ind/integration/fedot/extensions/catalog.json`. Регистрация выполняется
через `industrial_extension_scope()` или долгоживущую
`IndustrialExtensionSession`. Вложенные области и несколько экземпляров API в
одном контексте совместно владеют регистрацией: она снимается только после
закрытия последнего владельца.

Старый класс `IndustrialModels`, подмена внутренних реестров FEDOT и карта
перенаправления импортов удалены. Новый код не должен:

- вызывать `setup_repository()` или `setup_default_repository()`;
- менять `OperationTypesRepository.__repository_dict__`;
- подменять `PipelineSearchSpace.get_parameters_dict`;
- импортировать прежние модули `fedot.core.data.data`,
  `fedot.core.data.data_split` и `fedot.core.data.multi_modal`.

Для Dask область регистрации создаётся внутри отложенной задачи. Регистрация в
родительском процессе не считается достаточной, поскольку реестр расширений
FEDOT хранится в `ContextVar`.

Каталог явно задаёт `runtime_interface` каждой операции. Значение `array`
используется для моделей, принимающих массивы, а `input_data` — для реализаций,
работающих с `InputData`. Отложенная фабрика преобразует вход только согласно
этому объявлению; неявное угадывание интерфейса по исключению запрещено.

## Проверки

```bash
python -m tools.platform_support check --json
python -m tools.platform_support export --check
uv lock --check --python 3.10
python -m tools.fedot_import_boundary
python -m pytest tests/unit/platform_support -q
python -m pytest tests/unit/integration/fedot -q
python -m pytest tests/integration/fedot/test_regression_runtime.py -q
```

Для установленного wheel используется изолированный режим Python:

```bash
python -I tools/runtime_smoke.py --json
```

Проверка подтверждает, что импортируется установленный пакет, ресурсы реестра
доступны, FEDOT предоставляет новый контракт данных, а короткий сценарий
регрессии выполняет обучение и прогноз. Сообщения о ходе проверки выводятся в
stderr, итоговый JSON — в stdout.

## GitHub Actions

- `platform_checks.yml` проверяет метаданные на Python 3.10–3.12, установку
  wheel на Linux и Windows и интеграцию с FEDOT на Python 3.10/3.11;
- `poetry_unit_test.yml` использует `uv.lock` и запускает полный набор модульных
  тестов на Python 3.10;
- `integration_tests.yml` устанавливает тот же набор зависимостей и запускает
  интеграционные тесты;
- `package_build.yml` остаётся единственной процедурой сборки wheel и sdist.

Публикация в PyPI заблокирована, пока FEDOT указан прямой Git-ссылкой: PyPI не
принимает такие зависимости. Перед выпуском нужно заменить ссылку на
совместимую опубликованную версию FEDOT, обновить `uv.lock` и повторить все
проверки установки.

Подробности завершения перехода приведены в
`docs/dev_guide/ind_fedot_05_verification.md`.
