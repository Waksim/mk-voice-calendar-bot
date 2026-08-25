# Server deployment

The production container joins the existing `zozh-prod_ingress` network. Caddy
proxies only the exact Telegram webhook path to the bot. Because the VPS network
filters Telegram source ranges, a supervised Cloudflare Quick Tunnel provides
the actual inbound delivery route. Its controller checks the tunnel end to end
and re-registers Telegram whenever the ephemeral hostname changes. The direct
domain route remains available as a fallback for non-filtered clients.

This account-less Quick Tunnel is a transitional workaround, not an SLA-backed
ingress. Replace it with a Cloudflare named tunnel or another stable relay when
an account/token or an unfiltered public endpoint is available.

Runtime state is stored in `/srv/mk-voice-calendar-bot/runtime`. Secret files
are mounted read-only from `/etc/mk-voice-calendar-bot/secrets`. Required values
must use mode `0600`:

- `telegram-bot-token`
- `openrouter-api-key`
- `gemini-api-key`
- `telegram-api-id`
- `telegram-api-hash`
- `telegram-session-work`
- `telegram-session-personal`
- `telegram-personal-user-id`
- `telegram-work-user-id`
- `telegram-webhook-secret`

The bot reads provider credentials only through
`OPENROUTER_API_KEY_FILE=/run/secrets/openrouter-api-key` and
`GEMINI_API_KEY_FILE=/run/secrets/gemini-api-key`. Both secrets are required for
the complete failover chain. Do not put either key in Compose, an image layer,
or the release checkout.

At least one of the two `telegram-session-*` files must contain a dedicated
production `StringSession`; the other may be absent or empty while that account
is disabled. Never reuse a production value in a local Telegram MCP/Telethon
process: Telegram treats one auth key used concurrently from different IPs as
duplicated and permanently revokes that session.

Calendar planning uses a strict ordered provider chain:

1. OpenRouter `nvidia/nemotron-3-super-120b-a12b:free`, reasoning effort
   `medium`, stage timeout 35 seconds;
2. OpenRouter `z-ai/glm-5.2:free`, reasoning effort `high`, stage timeout 15
   seconds;
3. direct Gemini API `gemini-3.7-flash`, stage timeout 25 seconds.

Each planner invocation also has an 80-second overall deadline. A later stage
receives at most the time left under that deadline. The two free OpenRouter
stages deliberately use no same-model retries: timeout, `408`, `429`, `5xx`, an
unavailable route, or invalid structured output advances immediately to the
next stage. This prevents a throttled free route from consuming the whole
request budget before Gemini can run.

Free OpenRouter routes provide no capacity or latency guarantee. They may have
low rate limits, return `429`, lose an eligible upstream provider, or change
availability. They do not require positive paid OpenRouter credit, but the API
key must remain valid and its own optional usage limit must not be exhausted.
Direct Gemini is therefore an operational fallback, not an unused rollback
credential.

Defaults can be overridden with `OPENROUTER_MODEL`,
`OPENROUTER_REASONING_EFFORT`, `OPENROUTER_TIMEOUT_SECONDS`,
`OPENROUTER_FALLBACK_MODEL`, `OPENROUTER_FALLBACK_REASONING_EFFORT`,
`OPENROUTER_FALLBACK_TIMEOUT_SECONDS`, `OPENROUTER_MAX_TOKENS`, `GEMINI_MODEL`,
`GEMINI_TIMEOUT_SECONDS`, and `CALENDAR_PLANNER_TIMEOUT_SECONDS`. Startup
performs read-only capability checks for every configured stage. A permanently
rejected earlier stage is disabled for that process; a stage that only timed
out or hit a transient provider limit remains eligible and is retried first on
the next command. Startup fails if no planner stage validates. Webhook mode also
fails closed if the direct Gemini terminal stage fails credential/model checks;
a transient Gemini validation failure remains eligible for the next request.

The bot, outbound API proxy (the legacy `gemini-proxy` service name), and
webhook controller share the WARP network namespace. They use
`/etc/mk-voice-calendar-bot/resolv.conf` with explicit public resolvers; this
avoids Docker's embedded DNS becoming unavailable after namespace recreation.
All inter-sidecar traffic stays on loopback.

Build and start from a release directory:

```sh
sudo docker compose -f deploy/server/compose.yaml up -d --build
```

## GitHub CI/CD

Pushes to `main` run the locked test suite and then send only
`deploy <commit-sha>` through a restricted SSH key. The server fetches that
exact commit using a separate read-only GitHub deploy key. The bot image is
tagged with the commit SHA, while WARP and the webhook tunnel use their Git tree
IDs. App-only pushes therefore keep the established WARP session and tunnel
hostname. The deploy waits for all health checks, verifies Telegram's webhook,
and atomically switches `/opt/mk-voice-calendar-bot/current`.

Production uses a root-owned copy of `compose.yaml` from
`/etc/mk-voice-calendar-bot/compose.yaml`. A commit that changes Compose is
deliberately rejected until an administrator reviews and installs the new
manifest. This prevents an application-only push from adding host mounts,
capabilities, or privileged containers. Failed releases actively restore the
previous SHA-tagged images with `--no-build`; persistent runtime data is never
rolled back or deleted.

The application starts its authenticated HTTP listener without writing Telegram
configuration. Once healthy, the tunnel controller registers the active tunnel
hostname without dropping pending updates and keeps it current. This single
writer avoids a stale direct URL racing the controller after bot restarts.
Telegram deliveries are atomically persisted before returning HTTP 200 and
processed serially by a background worker.
