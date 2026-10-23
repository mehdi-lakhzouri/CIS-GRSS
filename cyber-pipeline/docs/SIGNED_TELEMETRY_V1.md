# Signed telemetry v1 and `sensorlab-c14n-1`

This specification describes the exact local signing contract implemented by SensorLab.
It is a semantic encoding of validated telemetry, **not RFC 8785/JCS**, and does not sign
the original NDJSON whitespace. The public verification vector is
[`signed-telemetry-v1-vector.json`](../validation/signed-telemetry-v1-vector.json).

## Envelope

The signed JSON object has exactly these required members:

| Member | Value |
|---|---|
| `envelope_version` | String `1.0` |
| `algorithm` | String `Ed25519` |
| `serialization` | String `sensorlab-c14n-1` |
| `key_id` | `ed25519-` followed by lowercase SHA-256 hex of the raw 32-byte public key |
| `telemetry` | The complete, unchanged unsigned TelemetryMessage 1.0 object |
| `signature` | Canonical standard base64 of the 64-byte Ed25519 signature |

No public key, private key, certificate, or password is embedded. The trusted public key is
selected by the telemetry sensor ID and envelope key ID in an operator-supplied registry.
The station must agree with both the fixed sensor map and the registry entry. Its status
must be `active`; `revoked` keys fail. Unknown keys/sensors fail. No algorithm negotiation
or fallback occurs.

Both the envelope and telemetry schemas forbid extra fields. Every one of the 19
measurement keys must be present, as a finite numeric value or JSON null. Parsing rejects
duplicate members at every level, NaN, Infinity, invalid signatures/base64, and invalid
telemetry. Registry parsing also rejects duplicate members and unknown fields.

## Exact bytes signed

1. Validate telemetry against TelemetryMessage 1.0. Decode its finite measurements as
   IEEE-754 binary64 values, session as a UUID, and timestamps as UTC datetimes. Integer,
   identity, and measurement type constraints still apply; numeric strings are rejected.
2. Make a signing-only copy of **all eight telemetry fields**: `schema_version`, `sensor_id`,
   `station`, `session_id`, `sequence`, `observation_time`, `sent_at`, and `measurements`.
3. Encode `session_id` in the UUID's lowercase hyphenated form. Encode the two timestamps as
   `YYYY-MM-DDTHH:MM:SS.ffffffZ`, always with six fractional digits. `observation_time` must
   be exactly on the hour. Sequence remains a positive decimal JSON integer.
4. In this signing copy only, encode each non-null measurement as the lowercase string
   returned by Python 3.12+ `float.hex()` for its binary64 value. Examples:

   | Numeric value | Signing representation |
   |---|---|
   | `1.0` | `"0x1.0000000000000p+0"` |
   | `0.5` | `"0x1.0000000000000p-1"` |
   | `0.0` | `"0x0.0p+0"` |
   | `-0.0` | `"-0x0.0p+0"` |
   | Smallest positive subnormal | `"0x0.0000000000001p-1022"` |
   | Missing | `null` |

   Finite nonzero normal values have one leading hexadecimal digit, a point, 13 fractional
   hexadecimal digits, `p`, and a signed decimal exponent with no unnecessary leading zeros.
   Subnormals use leading `0` and exponent `-1022`. Zero uses the special spelling above.
   Numeric values and null are **not** converted to these strings in transmitted telemetry.
5. Construct an object with exactly `envelope_version`, `algorithm`, `serialization`,
   `key_id`, and the signing copy of `telemetry`. The first three values are the envelope
   constants above. The `signature` member is excluded.
6. Serialize with recursive lexicographic sorting of object keys, ASCII JSON escaping,
   separators `,` and `:` without whitespace, and non-finite values forbidden. This is
   `json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)`.
   All supported nonnumeric strings are ASCII; keys are the fixed schema keys.
7. UTF-8 encode the JSON and prepend this exact domain separator:

   ```python
   b"SENSORLAB-SIGNED-TELEMETRY\x00v1\x00"
   ```

8. Apply standard Ed25519 directly to these bytes using `cryptography`. There is no external
   prehash, Ed25519ph mode, newline, or byte-order mark. Base64 encode the resulting 64 bytes.

These rules eliminate dependence on decimal float formatting in signature inputs and
retain the distinction between positive zero, negative zero, and null. JSON member order,
insignificant whitespace, equivalent UUID spellings, equivalent decimal binary64 values,
and equivalent UTC timestamp precision may normalize to the same signed meaning. They are
not treated as tampering. Any change to the normalized authenticated values fails.

## Verification

The verifier strictly parses and validates the envelope, looks up the sensor/key in the
trusted registry, rejects revoked entries, reconstructs the bytes above, and calls
`Ed25519PublicKey.verify(signature, bytes)`. It never unlocks or reads private keys. The
public interop vector includes the exact signing bytes in base64, public key, complete
envelope and signature for independent implementations to check.

The registry itself is a local trust anchor. Obtain it through a trusted operator channel,
keep it protected against unauthorized edits, and pass its path explicitly when verifying
outside the source directory. A public key in an untrusted message cannot establish trust.

Registry/signing objects take a snapshot when loaded. Revocation affects subsequent loads;
live registry synchronization and key rotation are future work. A newly revoked key also
fails verification of older envelopes under that current registry policy.

## Security scope

Signatures authenticate possession of a registered private key and bind the fields listed
above. They do not assert that an environmental observation is attack-free or physically
correct. Historical observations can already include the exported FDI values.

Verification is intentionally stateless: an identical valid envelope verifies repeatedly.
Session authorization, permitted time windows, persisted sequence tracking, replay rejection,
network identity, transport confidentiality, and HTTP ingestion belong to the future Secure
Ingestion API. They are not implemented here.
