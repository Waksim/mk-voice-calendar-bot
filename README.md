# Telegram Voice/Text/Image → Google Calendar

Бот `@mk_voice_text_bot` только для владельца и двух его аккаунтов. Он принимает
обычную текстовую календарную команду, голосовое либо скриншот/изображение через
Bot API. Голосовое
сопоставляется с исходящим сообщением в выбранной пользовательской сессии и
расшифровывается вызовом Telegram MTProto `messages.transcribeAudio`; файл не
скачивается, распознавание выполняется на серверах Telegram. Текст сразу
переходит к планированию без Telegram user session. Planner вызывает провайдеров
в фиксированном порядке: subscription-backed Codex CLI `gpt-5.6-sol`,
бесплатный `nvidia/nemotron-3-super-120b-a12b:free` через OpenRouter,
бесплатный `z-ai/glm-5.2:free` через OpenRouter, direct Gemini API и последним
direct API `GigaChat-2-Max` с принудительным вызовом функции и строгой схемой
аргументов. Эта цепочка
планирует создание, изменение или удаление календарных событий. Однозначные
команды чтения вроде «покажи события в
ближайший час» разбираются детерминированно без LLM. Операция сразу выполняется
в основном Google Calendar, а финальная карточка содержит кнопку отмены. Для
удаления событий с расширенными provider-метаданными отмена восстанавливает
основные календарные поля, но может не вернуть гостей, цвет, конференцию или
нестандартные напоминания.

Codex получает не более 55 секунд. GigaChat получает не более 45 секунд и
делает не более одного ограниченного
повтора при `429`/`5xx`. У OpenRouter-ступеней нет внутренних повторов: тайм-аут,
`408`, `429`, `5xx`, отказ маршрута или ответ вне схемы сразу передают тот же
запрос следующей модели. Nemotron получает не более 35 секунд, GLM — 15 секунд,
прямой Gemini — 25 секунд; весь один вызов planner дополнительно ограничен общим
дедлайном 180 секунд, поэтому поздняя ступень получает только оставшееся время.
Ошибки авторизации и лимитов показываются как недоступность провайдера, а не как
ошибка разбора календарной команды.

К каждому новому запросу модель получает максимум два последних диалоговых хода
и одно компактное, нормализованное представление каждого релевантного события.
Нерелевантный small talk не занимает эти слоты: потерянные в старом state
последние календарные действия восстанавливаются из durable operation journal.
Google event IDs остаются на сервере; модель видит короткие ссылки `e1`, `e2` и
серверную allowlist. Старые user input, application state, model output и
непрозрачные thought signatures между пользовательскими командами не
пересылаются. Поэтому продолжения вроде «добавь переговорную А», «перенеси
планёрку» и «удали это» адресуются к фактически созданному событию, а не
создают замену. Модель возвращает update как patch: неупомянутые дата, время,
длительность и остальные поля сохраняются. Правило повторения тоже можно
заменить или очистить; такое изменение адресуется к master-событию всей серии,
при этом известные `EXDATE`/`RDATE` сохраняются. Для повторяющегося события
модель явно выбирает всю серию или конкретный экземпляр; если из команды это
непонятно, бот задаёт уточняющий вопрос.

Для событий, которых ещё нет в локальном журнале, update/delete выполняется в
два прохода. Модель сначала задаёт узкое окно поиска, приложение читает
кандидатов через Calendar MCP и только затем повторно вызывает planner с
точной allowlist коротких ссылок на найденные события. Контекст первого прохода
сохраняется только внутри этого одного хода `план → lookup → выбор`. Неполная
или неоднозначная выборка не останавливается отдельным policy-gate: приложение
передаёт доступных кандидатов и признак неполноты второму вызову модели, а она
выбирает CRUD-действие либо задаёт уточняющий вопрос. Команды «покажи» и «найди»
используют тот же bounded lookup как read-only операцию без кнопки отмены.

Непосредственное продолжение по `e1`/`e2` не запускает новый list/search и не
требует второго вызова модели. Перед каждым update/delete приложение всё же
читает точное событие у провайдера: этот свежий снимок становится основой patch
и Undo и не позволяет локальному кэшу затереть изменение, сделанное вручную в
Google Calendar. Проверка не отменяет валидный CRUD-план модели из-за
confidence, повторения, гостей, конференции, вложений, нестандартных
напоминаний или специального типа события.
Перед компенсирующей записью Undo адаптер ещё раз условно сверяет полный снимок
события. Ручная правка, попавшая между предварительным чтением и записью,
блокирует Undo, а не перезаписывается.
Неоднозначный результат записи до пяти раз повторяется с тем же journal и
idempotency key, поэтому рестарт или потерянный ответ провайдера не создают
новый batch.

Для timed recurring events адаптер передаёт локальное wall-clock время вместе
с IANA timezone: pinned MCP 2.6.2 иначе отбрасывает timezone у RFC3339-значения
с offset, а Google требует его для разворачивания серии через DST. Однозначный
provider reject подтверждается чтением детерминированного ID и не расходует
пять попыток, предназначенных только для действительно неопределённых записей.

Запись выполняет локальный stdio MCP subprocess
`@cocal/google-calendar-mcp` версии `2.6.2`. Логические Telegram-аккаунты
`personal` и `work` оба направляются в авторизованный MCP-аккаунт `owner` и
календарь `primary`.

Первый текстовый planner — Codex CLI с авторизацией ChatGPT-подписки и моделью
`gpt-5.6-sol`. CLI работает в отдельном read-only sidecar без Telegram,
Google Calendar и provider-секретов. Бот передаёт ему по защищённому локальному
RPC только готовый prompt одного из двух фиксированных типов; модель, reasoning
effort и JSON Schema задаются самим runner. Запуск сериализован, не использует
постоянные треды, правила репозитория или инструменты. Сессия лежит вне Git и
образа в отдельном writable `CODEX_HOME`. Детали официального входа и
non-interactive режима: [Codex authentication](https://developers.openai.com/codex/auth)
и [Codex non-interactive mode](https://developers.openai.com/codex/noninteractive).

Последний API-фолбэк — direct API `GigaChat-2-Max`. Бот получает
короткоживущий OAuth-токен по `GIGACHAT_CREDENTIALS`, кэширует его и проверяет
TLS официальным корневым сертификатом НУЦ Минцифры из
`deploy/server/certs/russian_trusted_root_ca_pem.crt`. В Git credential и
access token не сохраняются. Формат интеграции следует официальной документации
по [авторизации](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api),
[вызову функций](https://developers.sber.ru/docs/ru/gigachat/guides/functions/generating-arguments-for-custom-functions)
и [сертификатам](https://developers.sber.ru/docs/ru/gigachat/certificates).

Резервные OpenRouter-модели используют бесплатные `:free`-маршруты. Бесплатность
не означает гарантированную доступность: OpenRouter может применять низкие
rate limits, возвращать `429`, временно не иметь подходящего upstream-провайдера
или изменить доступность модели. Gemini API остаётся независимой четвёртой
ступенью, а GigaChat — пятой; для полноценной production-цепочки нужны Codex ChatGPT-сессия и
секреты `CODEX_RUNNER_TOKEN`, `GIGACHAT_CREDENTIALS`, `OPENROUTER_API_KEY` и
`GEMINI_API_KEY`. Положительный платный баланс
OpenRouter для `:free`-моделей не требуется, однако API-ключ должен быть
валидным и не иметь исчерпанного собственного лимита.

Параметры цепочки по умолчанию:

- Codex: `CODEX_MODEL=gpt-5.6-sol`,
  `CODEX_REASONING_EFFORT=medium`, `CODEX_TIMEOUT_SECONDS=55`;
- Nemotron: `OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free`,
  `OPENROUTER_REASONING_EFFORT=medium`,
  `OPENROUTER_TIMEOUT_SECONDS=35`;
- GLM: `OPENROUTER_FALLBACK_MODEL=z-ai/glm-5.2:free`,
  `OPENROUTER_FALLBACK_REASONING_EFFORT=high`,
  `OPENROUTER_FALLBACK_TIMEOUT_SECONDS=15`;
- Gemini: `GEMINI_MODEL=gemini-3.7-flash`,
  `GEMINI_TIMEOUT_SECONDS=25`;
- GigaChat: `GIGACHAT_MODEL=GigaChat-2-Max`,
  `GIGACHAT_SCOPE=GIGACHAT_API_CORP`, `GIGACHAT_TIMEOUT_SECONDS=45`;
- общий предел одного planner-вызова:
  `CALENDAR_PLANNER_TIMEOUT_SECONDS=180`.

Для обеих OpenRouter-ступеней общий потолок ответа задаётся
`OPENROUTER_MAX_TOKENS` (по умолчанию 8192). Stage timeout ограничивает один
сетевой вызов, а общий planner timeout — всю последовательность fallback.

## Скриншоты и изображения

Изображение проходит отдельный observation-only конвейер. Vision-модели не
выделяют `start`, `end`, адрес или готовую календарную операцию: они возвращают
только нейтральное описание и максимально буквальную расшифровку видимого
текста. Эти два поля передаются в уже существующий planner как недоверенное
содержимое изображения вместе с подписью пользователя, если она есть. Решение о
CRUD, датах, времени и намерении пользователя по-прежнему принимает текстовая
цепочка Codex Sol → Nemotron → GLM → Gemini → GigaChat 2 Max.

Vision-провайдеры вызываются по одному, без общего дедлайна между ступенями:

1. OpenRouter `google/gemma-4-31b-it:free`, timeout 15 секунд;
2. OpenRouter `google/gemma-4-26b-a4b-it:free`, timeout 12 секунд;
3. direct Gemini API `gemini-3.7-flash`, timeout 20 секунд;
4. обязательный локальный RapidOCR 3 / ONNX Runtime с PP-OCRv5 Cyrillic и
   резервным ESLAV, timeout 15 секунд.

Локальная ступень является терминальным OCR-fallback: она восстанавливает
видимый текст, но не придумывает семантическое описание. Production-образ заранее
загружает обе кириллические модели в `/opt/rapidocr-models` и делает каталог
read-only, поэтому контейнеру не нужен доступ к сети или writable `$HOME` для
первого OCR-вызова.

По умолчанию принимается не более 8 MiB encoded image и 20 миллионов пикселей.
Сохранённый результат распознавания ограничен 4000 символами описания и 12000
символами видимого текста. Непосредственно перед planner эти поля получают ещё
более строгий общий бюджет: 1536 и 6656 UTF-8 байт соответственно. Это оставляет
место для кандидатов и истории внутри жёсткого лимита planner-запроса 64 KiB;
длинный OCR-текст завершается явной отметкой о сокращении. Исходные байты
изображения planner не получает. Пределы распознавания и timeouts настраиваются
через `OPENROUTER_VISION_MODEL`,
`OPENROUTER_VISION_TIMEOUT_SECONDS`, `OPENROUTER_VISION_FALLBACK_MODEL`,
`OPENROUTER_VISION_FALLBACK_TIMEOUT_SECONDS`, `GEMINI_VISION_MODEL`,
`GEMINI_VISION_TIMEOUT_SECONDS`, `VISION_LOCAL_OCR_TIMEOUT_SECONDS`,
`VISION_MAX_IMAGE_BYTES`, `VISION_MAX_IMAGE_PIXELS`,
`VISION_MAX_DESCRIPTION_CHARS`, `VISION_MAX_VISIBLE_TEXT_CHARS` и
`VISION_OCR_MODEL_DIR`.

## Безопасность

- Production-секреты не хранятся в GitHub. Они монтируются read-only из
  `/etc/mk-voice-calendar-bot`; runtime и OAuth state живут отдельно в
  `/srv/mk-voice-calendar-bot/runtime`.
- GigaChat credential, ключи OpenRouter и Gemini читаются из environment/secret
  files и передаются только в заголовках провайдерских HTTPS-запросов; они не
  попадают в URL или сообщения об ошибках. GigaChat access token хранится только
  в памяти процесса.
- Telegram user sessions получает из файлов только изолированный gateway;
  приложение обращается к нему как к MCP subprocess.
- Calendar MCP запускается с allowlist только из `create-event`, `get-event`,
  `list-events`, `search-events`, `update-event`, `delete-event` и
  `list-calendars`; его stderr не попадает в логи сервиса.
- Каждому создаваемому событию назначается детерминированный Google event ID.
  Создания, изменения, удаления и их отмена записываются в atomic journal с
  before/after snapshots. Повтор update, callback или рестарт после потерянного
  ответа не должен повторно изменить календарь.
- Доступ разрешён только двум numeric Telegram user ID из локального `.env`
  или read-only production secret files; сами ID в Git не хранятся.
- Тексты расшифровок не пишутся в service logs. Две последние пары хранятся в
  journal с правами `0600`; модель получает только их компактное
  структурированное представление и короткие ссылки на события.
- Байты изображений, их description и OCR-текст не пишутся в service logs.
  Логи Vision содержат только имя провайдера/модели, размеры, длительность и
  тип безопасной ошибки.

## Разработка и деплой

Production работает на VPS в Docker Compose и принимает Telegram updates через
webhook. Push в `main` запускает GitHub Actions: locked-тесты, валидацию
deployment-файлов и затем ограниченный SSH-триггер. Сервер сам забирает точный
commit read-only deploy key, собирает content-addressed образы, ждёт health checks и
при ошибке возвращает предыдущие образы без отката persistent state.

Локальная проверка:

```sh
cp .env.example .env
chmod 600 .env
# Заполните оба Telegram user ID, CODEX_RUNNER_TOKEN, GIGACHAT_CREDENTIALS,
# OPENROUTER_API_KEY и GEMINI_API_KEY.
uv sync --locked --group dev
uv run pytest -q
docker compose -f deploy/server/compose.yaml config --quiet
```

Production Compose хранится на сервере как root-owned файл. Его изменение в
GitHub намеренно требует отдельного административного одобрения, тогда как
обычные изменения Python-кода и Docker build-контекста деплоятся автоматически.
Подробности — в `deploy/server/README.md`.
