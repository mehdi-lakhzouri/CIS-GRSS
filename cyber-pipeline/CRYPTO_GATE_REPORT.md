# Crypto Gate — Stage 1 Secure Ingestion

Completed **23 October 2026**. Flow demonstrated end to end:
**Sensor Emulator → Ed25519 signing → verified HTTPS → Crypto Gate → persisted security decision.**
Stage 2 (AI detection, ERA5, dashboard) is **not** implemented; accepted observations are only
written to a durable `accepted_observations` table (the Stage 2 outbox) with no consumer.

## Architecture (`src/sensorlab/gate/`)

| Module | Role |
|---|---|
| `api.py` | FastAPI app: `POST /v1/telemetry`, `GET /health`; HTTPS-only, streaming size cap, no docs/OpenAPI, no network admin endpoints |
| `models.py` | Pydantic strict ingress schema (`extra="forbid"`), `GateConfig`, `Decision` |
| `service.py` | `CryptoGate` ordered pipeline; reuses `TrustedRegistry` and `verify_envelope` (no second signing implementation) |
| `repository.py` | `GateRepository` abstraction + `SQLiteGateRepository` (sessions, last sequences, decisions, outbox) |
| `http_transport.py` | `TelemetrySink` HTTPS adapter for SensorLab |
| `tls.py` | Development CA/localhost certificate generation |
| `cli.py` | `sensorlab gate …` operator commands |

## Verification order

1. **E_SCHEMA** – strict JSON (duplicate members rejected), strict envelope, no extra fields.
2. **E_UNKNOWN_KEY / E_REVOKED_KEY** – sensor, station and `key_id` must match the operator registry,
   which is reloaded per request (revocation is immediate). Request-supplied keys are never used.
3. **E_BAD_SIG** – Ed25519 over `sensorlab-c14n-1` via the existing verifier.
4. **E_UNAUTHORIZED_SESSION** – session must be operator-authorized for that sensor/key and inside
   its `not_before`/`expires_at` window and not revoked. New UUIDs are never auto-trusted.
5. **E_REPLAY_SEQ** – sequence must be strictly greater than the persisted last sequence; gaps are
   accepted with `W_SEQ_GAP` and the gap size recorded.
6. **E_STALE** – signed `sent_at` vs. server receive time (default max age 120 s, future skew 5 s).
   `observation_time` is never used. A stale packet does not consume its sequence.
7. **Accept** – in one SQLite transaction: sequence advance, audit record and `accepted_observations` row.
   If the outbox write fails the whole transaction rolls back (tested).

Other codes: `E_PAYLOAD_TOO_LARGE` (413, default 16 KiB), `E_HTTPS_REQUIRED` (spoofed
`X-Forwarded-*` is ignored), `E_REGISTRY`/`E_STORAGE` (503, fail closed). Unsigned messages are
rejected at the endpoint with `E_SCHEMA`; unsigned/NDJSON output of SensorLab is unchanged.

**No nonce exists in the signed envelope and none was added.** Replay protection is
session + per-sensor sequence + `sent_at` freshness only. A nonce requires a new, fully signed
protocol version.

## Persistence and concurrency

SQLite (WAL) stores authorized sessions, last sequences and decisions. Admission uses an
immediate write transaction so concurrent duplicates serialize; only one is accepted. State
survives restart, and re-authorizing an existing session cannot reset its sequence. Sessions are
created/expired/revoked only through the local operator CLI (no network admin interface).

## Audit logging

Each decision stores timestamp, sensor, station, session, sequence, decision, code, reason,
warnings, gap and packet SHA-256 (`digest_scope` says whether over the full body or only the
received prefix). Raw payloads, measurements, signatures, keys and passwords are never logged;
metadata from unverified requests is regex/UUID-validated and bounded before recording.

## CLI

```bash
uv run sensorlab gate dev-cert --directory .sensorlab/tls
uv run sensorlab gate authorize <UUID> --expires-at <ISO-UTC> [--sensor-id ID]… [--not-before ISO]
uv run sensorlab gate serve --cert .sensorlab/tls/server.pem --key .sensorlab/tls/server.private.pem \
    [--host 127.0.0.1 --port 8443 --max-payload-bytes N --max-age-seconds S --future-skew-seconds S]
uv run sensorlab replay --signed --session-id <UUID> --endpoint https://127.0.0.1:8443/v1/telemetry \
    --ca-cert .sensorlab/tls/ca.pem [--speed 3600 --start … --end …]
uv run sensorlab gate status | events [--limit N] | revoke-session <UUID>
uv run python scripts/validate_crypto_gate.py
```

`dev-cert` creates a seven-day CA and localhost/127.0.0.1 certificate in a new restricted,
Git-ignored directory and changes no system trust store; clients must pass `--ca-cert`.
Certificate verification cannot be disabled (tested). Passwords use hidden prompts or the
`SENSORLAB_KEY_PASSWORD` / `SENSORLAB_TLS_PASSWORD` environment variables. `--session-id`
reuses the authorized UUID; sequence and timestamp semantics are unchanged.

## Verified results

Real combined USCRN 24-hour release (`2025-01-01T01:00Z` – `2025-01-02T01:00Z`), all six
sensors, 3600×, verified HTTPS, disposable encrypted keys and a fresh database
(`validation/crypto_gate_results.json`):

| Check | Result |
|---|---|
| Messages transmitted / accepted / rejected | **144 / 144 / 0** (24 per sensor, sequences 1–24) |
| Outbox rows after replay | 144 |
| Replay of an identical packet | `E_REPLAY_SEQ` |
| Tampered measurement | `E_BAD_SIG` |
| Sequence gap (26 after 24) | ACCEPT with `W_SEQ_GAP`, gap 1 |
| Out-of-order sequence | `E_REPLAY_SEQ` |
| Unauthorized session | `E_UNAUTHORIZED_SESSION` |
| Stale / future `sent_at` | `E_STALE` ×2 |
| Wrong/unknown key; revoked key | `E_UNKNOWN_KEY`; `E_REVOKED_KEY` |
| Final accounting | 146 accepted (`OK`), 9 rejected, 1 gap event, 146 outbox rows — consistent |
| API restart | Previously accepted sequences remained protected |
| Source data | SHA-256 of input Parquet and contract unchanged (`ff478b38…`, `bbee3a73…`) |

Quality gates: **168 tests passed** (130 pre-existing SensorLab tests plus 38 new gate tests,
including a real 144-message HTTPS test and concurrent-duplicate/restart tests); `ruff check`
and `ruff format --check` clean. One third-party Starlette/httpx deprecation warning remains.

```bash
uv run pytest -q --basetemp=.pytest-tmp-gate-final
uv run ruff check . ; uv run ruff format --check .
```

## Limitations

- Trust is local: the registry file is the trust anchor and must be protected by the operator.
  It is re-read on each request; there is no signed/remote registry distribution.
- Single host, SQLite. The repository interface allows PostgreSQL, but none is implemented.
- Freshness depends on synchronized clocks and a trusted server clock; within the freshness
  window, replay is stopped only by sequence state. Without a nonce, a captured packet whose
  sequence has *not yet* been consumed (for example one delayed or withheld by an attacker)
  can still be delivered while fresh.
- The gate authenticates the sender and message integrity, not data truth; FDI attacks by a
  holder of valid keys are Stage 2's concern.
- Development CA/certificates are for local use only; no mTLS, rate limiting, or DoS controls
  beyond payload size limits. Production needs a reverse proxy/WAF and managed certificates.
- Stage 2 outbox has no consumer or retention policy; audit data is not tamper-evident or rotated.
- Validated on Windows only.
