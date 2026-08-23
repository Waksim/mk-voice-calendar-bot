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
- `gemini-api-key`
- `telegram-api-id`
- `telegram-api-hash`
- `telegram-session-work`
- `telegram-session-personal`
- `telegram-personal-user-id`
- `telegram-work-user-id`
- `telegram-webhook-secret`

At least one of the two `telegram-session-*` files must contain a dedicated
production `StringSession`; the other may be absent or empty while that account
is disabled. Never reuse a production value in a local Telegram MCP/Telethon
process: Telegram treats one auth key used concurrently from different IPs as
duplicated and permanently revokes that session.

The bot, Gemini proxy, and webhook controller share the WARP network namespace.
They use `/etc/mk-voice-calendar-bot/resolv.conf` with explicit public resolvers;
this avoids Docker's embedded DNS becoming unavailable after namespace
recreation. All inter-sidecar traffic stays on loopback.

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
