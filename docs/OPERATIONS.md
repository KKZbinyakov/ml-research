# Эксплуатация локального стенда

Документ описывает повторный запуск, обновление и диагностику стенда из README. Для первого знакомства начните с раздела «Локальный запуск» в [README](../README.md).

## Состав и требования

Стенд запускает vLLM 0.27.0 CPU в kind, отдельный initContainer проверяет и при необходимости загружает модель на PVC. Основной контейнер работает с локальной копией модели в offline-режиме. По умолчанию используется `Qwen/Qwen2.5-0.5B-Instruct` на закреплённой ревизии `7ae557604adf67be50417f59c2c2f167def9a775`.

Нужны Docker с Linux-контейнерами, Python 3.12+, `kubectl`, интернет для первичной загрузки образов и весов. Для локальной конфигурации README рекомендует Docker Desktop выделить 12 ГиБ RAM, 6 логических CPU и 12 ГиБ диска; Pod vLLM запрашивает 4 CPU и 4 ГиБ RAM, с лимитом памяти 7 ГиБ. PVC нужен с `ReadWriteOnce` и StorageClass по умолчанию.

`kind` версии `v0.33.0` обязателен. Скрипт принимает эту версию из PATH или из локального `.tools`; несовпадающую версию пропускает. Если закреплённой копии нет, он скачивает бинарник и файл SHA256, сверяет контрольную сумму и только после совпадения атомарно заменяет локальную копию. Поэтому в сеть скрипт обращается только при отсутствии проверенной копии.

## Команды

Запускайте из любого каталога:

```powershell
python C:\путь\к\ml_research\tools\local.py up
python C:\путь\к\ml_research\tools\local.py status
python C:\путь\к\ml_research\tools\local.py forward
python C:\путь\к\ml_research\tools\local.py verify
python C:\путь\к\ml_research\tools\local.py benchmark --label reduced-smoke --requests 8 --repeats 2 --concurrency 1,2,4 --workload both
```

Из каталога проекта вместо полного пути можно писать `python tools/local.py ...`. `up` создаёт кластер, собирает и загружает образы, применяет Kustomize и ждёт готовности Deployment; на повторном запуске экспортирует kubeconfig существующего кластера. Первый запуск initContainer может скачать примерно 1 ГБ весов. Нагрузка в `benchmark` отдельно прогревает каждую пару workload/concurrency, а затем исключает прогрев из измерения.

Параметры кластера и локального API-порта:

```powershell
python tools/local.py --cluster ml-research-acceptance up
python tools/local.py status --cluster ml-research-acceptance
python tools/local.py forward --cluster ml-research-acceptance --port 19000
python tools/local.py verify --cluster ml-research-acceptance --port 19000
python tools/local.py benchmark --cluster ml-research-acceptance --port 19000 --requests 8 --concurrency 1,2
python tools/local.py down --cluster ml-research-acceptance
```

Имя должно быть `ml-research` либо начинаться с `ml-research-`. Основной кластер использует `.tools/kubeconfig`; для другого скрипт выбирает `.tools/kubeconfig-{cluster}`. Можно передать `--kubeconfig PATH`. Токен каждого кластера хранится отдельно в `.secrets/api-token` или `.secrets/api-token-{cluster}`. `--port` используется переадресацией и клиентскими командами; при необходимости явный `--base-url` в аргументах `verify` или `benchmark` переопределит URL, построенный скриптом.

`tools/client.py` предназначен для кластера по умолчанию: он всегда обращается к `http://127.0.0.1:18000` и читает `.secrets/api-token`. Он не выбирает kubeconfig, кластер или порт по аргументам. Для другого кластера пользуйтесь `local.py verify` или `local.py benchmark`, где `--cluster` выбирает kubeconfig и токен, а `--port` задаёт URL.

`down` удаляет только указанный kind-кластер. PVC и его содержимое тоже исчезнут вместе с кластером. Файлы проекта и токены в `.secrets` команда не удаляет; удалите их отдельно, если действительно хотите их убрать.

## Конфигурация, образы и gated-модель

`k8s/kustomization.yaml` применяет Namespace, ConfigMap, PVC, Deployment и Service из `k8s/base`. Настройки модели и сервера находятся в `k8s/base/configmap.yaml`; после изменения выполните `kubectl apply -k k8s`, затем перезапустите Deployment и дождитесь rollout. Локальный скрипт делает это при `up`.

В `deployment.yaml` задано `enableServiceLinks: false`: Kubernetes иначе добавит переменную `VLLM_PORT` для Service, а vLLM использует имя `VLLM_PORT` в своей конфигурации. В `kind.yaml` `failCgroupV1: false` сохраняет совместимость локального kubelet с указанной установкой Docker Desktop; настройка относится только к этому kubelet.

`Dockerfile` наследуется от официального CPU-образа vLLM, закреплённого по SHA256 digest; CUDA-сборки здесь нет. Lock-файл содержит все 155 дистрибутивов базового образа и сборки проверяют точный состав вместе с `pip check`. `Dockerfile.init` использует slim Python-образ и 16 закреплённых пакетов HF-загрузчика. Оба контейнера запускают процессы под UID/GID 10001, с `allowPrivilegeEscalation: false`, всеми capabilities снятыми, `RuntimeDefault` seccomp и read-only root filesystem. Основной контейнер монтирует PVC только для чтения; временные файлы идут в `/tmp` и `/dev/shm`.

Для gated-модели задайте `HF_TOKEN` в окружении перед `up` и поменяйте ID модели и полный commit SHA в ConfigMap. Скрипт передаёт HF-токен только в initContainer через Secret. Для следующего `up` повторно задайте `HF_TOKEN`, поскольку Secret перезаписывается данными окружения. Не вставляйте значение в манифест или командную строку.

## PVC и восстановление кеша

Загрузчик сначала сверяет ID модели, ревизию, размеры и SHA256 каждого файла. Проверка также валидирует нужные JSON и заголовки safetensors. Маркер `.download-complete.json` должен содержать список хешей `sha256`; прежний маркер без этих хешей не считается готовым, поэтому загрузчик выполнит новую загрузку. Не удаляйте маркер вручную.

Кеш использует три каталога: `/models/model` (активные веса), `/models/model.staging` (новая загрузка) и `/models/model.previous` (резервная копия). После сбоя корректный staging можно продвинуть в активный каталог; если staging неполон, но backup цел, загрузчик восстановит backup. При старте он ещё раз проверяет все файлы, поэтому частичный каталог не выдаётся за готовую модель. Неудача новой загрузки не должна заменять валидные текущие веса.

Тёплый запуск не вызывает загрузчик HF: он проверяет кеш и завершается до импорта `huggingface_hub`. Затем сервер использует локальный путь модели, `HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`. Это поведение приложения, а не Kubernetes NetworkPolicy.

## Авторизация и журналы

Локальный API-токен создаётся в `.secrets` и хранится только на этой машине. `tools/client.py` читает его оттуда. В Kubernetes Secret передаётся через stdin `kubectl`; токен не включается в аргументы команды. Защита API требует один заголовок `Authorization: Bearer ...` на всех маршрутах, кроме `GET/HEAD /health`; сравнение выполняется в constant time. WebSocket закрывается.

Журнал не содержит текста запроса и ответа, query string, клиентских заголовков или исключений. Middleware пишет UUID, метод, маршрут из ограниченного набора, статус и длительность. `serve.py` перехватывает stdout/stderr дочернего vLLM: разрешённые записи проходят только после проверки HMAC и схемы. Ключ HMAC случайный для каждого запуска процесса. HMAC подтверждает целостность подписи, он не шифрует запись. Не прошедшие фильтр строки отбрасываются; из диагностики останутся категория компонента, стадия, тип ошибки из allowlist, код завершения и число отброшенных строк.

## Проверки и результаты

Модульные проверки запускаются стандартной библиотекой Python:

```powershell
python -m unittest discover -s tests -v
```

С работающим `forward` команда `verify` обращается к API и проверяет health, Bearer-авторизацию, генерацию, streaming, ограничение журналов и свойства работающего контейнера. Результат сохраняется в [artifacts/verification.json](../artifacts/verification.json). `python tools/verify_cache.py` перезапускает Pod, проверяет, что прогретый PVC обходится без HF SDK и Python-сокетов, а затем восстанавливает рабочую конфигурацию. После неё запустите `forward` заново.

`benchmark` пишет JSON в путь `--output`; без него это `bench/results.json`. Команда выше помечена как `reduced-smoke`. Эталонная полная серия использует `--requests 30 --repeats 2 --concurrency 1,2,4 --workload both`; не сравнивайте короткую smoke-серию с ней как равный объём измерений. `p95` на малом количестве запросов зависит от отдельного запроса и не служит оценкой production SLO.

Если Pod не становится Ready, сначала проверьте состояние и события:

```powershell
python tools/local.py status
kubectl --kubeconfig .tools/kubeconfig -n ml-research describe pod -l app=vllm
kubectl --kubeconfig .tools/kubeconfig -n ml-research logs deployment/vllm -c download-model
kubectl --kubeconfig .tools/kubeconfig -n ml-research logs deployment/vllm -c vllm
```

Фильтр намеренно не печатает текст ошибки. Используйте стадию и тип ошибки в разрешённом событии, состояние rollout и события Pod, не отключая защиту логов ради полного traceback.
