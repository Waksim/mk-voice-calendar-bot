# Telegram Voice → Google Calendar

Бот `@mk_voice_text_bot` только для владельца и двух его аккаунтов. Он получает
голосовое через Bot API, сопоставляет его с исходящим сообщением в выбранной
пользовательской сессии и вызывает Telegram MTProto
`messages.transcribeAudio`. Голосовой файл не скачивается; распознавание
выполняется на серверах Telegram. Затем Gemini Developer API вызывает
`gemini-3.7-flash` через Interactions API с high thinking и по строгой JSON
Schema планирует чтение, создание, изменение или удаление календарных событий.
Официальный Google Antigravity CLI остаётся резервным провайдером. Операция
сразу выполняется в основном Google Calendar, а финальная карточка содержит
кнопку отмены. Для удаления событий с расширенными provider-метаданными отмена
восстанавливает основные календарные поля, но может не вернуть гостей, цвет,
конференцию или нестандартные напоминания.

К каждому запросу Gemini получает два последних диалоговых хода, точные
stateless steps Interactions API и доверенный снимок фактически выполненных
операций с Google event IDs. Поэтому продолжения вроде «добавь переговорную
А», «перенеси планёрку» и «удали это» адресуются к уже созданному
событию, а не создают новую запись. Модель возвращает update как patch:
неупомянутые дата, время, длительность и остальные поля сохраняются.

Для событий, которых ещё нет в локальном журнале, update/delete выполняется в
два прохода. Gemini сначала задаёт узкое окно поиска, приложение читает
кандидатов через Calendar MCP и только затем повторно вызывает Gemini с точным
allowlist найденных Google event IDs. Неполная или неоднозначная выборка не
останавливается отдельным policy-gate: приложение передаёт доступных кандидатов
и признак неполноты второму вызову Gemini, а модель выбирает CRUD-действие либо
задаёт уточняющий вопрос. Команды «покажи» и «найди» используют тот же bounded
lookup как read-only операцию без кнопки отмены.

Перед update/delete приложение повторно читает выбранное событие у провайдера.
Свежий снимок становится основой patch и Undo, но не отменяет валидный CRUD-план
Gemini из-за confidence, повторения, гостей, конференции, вложений,
нестандартных напоминаний, специального типа или параллельного изменения полей.
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

## Безопасность

- Production-секреты не хранятся в GitHub. Они монтируются read-only из
  `/etc/mk-voice-calendar-bot`; runtime и OAuth state живут отдельно в
  `/srv/mk-voice-calendar-bot/runtime`.
- Gemini API key передаётся только в заголовке `x-goog-api-key`; он не попадает
  в URL или сообщения об ошибках.
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
  journal с правами `0600`, чтобы Gemini понимала продолжения.

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
# Заполните TELEGRAM_PERSONAL_USER_ID и TELEGRAM_WORK_USER_ID.
uv sync --locked --group dev
uv run pytest -q
docker compose -f deploy/server/compose.yaml config --quiet
```

Production Compose хранится на сервере как root-owned файл. Его изменение в
GitHub намеренно требует отдельного административного одобрения, тогда как
обычные изменения Python-кода и Docker build-контекста деплоятся автоматически.
Подробности — в `deploy/server/README.md`.
