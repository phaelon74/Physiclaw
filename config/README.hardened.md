# Hardened Config Example

`openclaw.hardened.example.json` is a Docker-first reference config with security hardening applied.

**Usage:** Copy to `openclaw.json` and set environment variables:

- `PHYSICLAW_GATEWAY_TOKEN` — gateway auth token
- `VLLM_API_KEY` — model provider API key (never hardcode in config)

**Key hardening settings:**

- `gateway.bind: "0.0.0.0"` — correct for Docker (container needs all interfaces for port mapping). For bare-metal, use `"loopback"` or `"127.0.0.1"`.
- `gateway.auth` — token required for all connections
- `tools.profile: "messaging"` — minimal tool set
- `tools.deny` — blocks high-risk tools (automation, runtime, session spawning)
- `tools.exec.security: "allowlist"` — only explicitly allowed binaries
- `tools.exec.ask: "always"` — require approval before running commands
- `logging.redactSensitive: "tools"` — redact secrets in logs

**Migrate existing configs:** Replace any inline API keys (e.g. vLLM `apiKey`) with env var references like `${VLLM_API_KEY}`.

**Workspace bootstrap:** When a new workspace is created (e.g. empty `./data` on first run), the gateway seeds AGENTS.md, SOUL.md, etc. from `docs/reference/templates/`. The strengthened safety rules in those templates are used automatically.
