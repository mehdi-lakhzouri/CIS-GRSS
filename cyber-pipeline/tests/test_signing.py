import io
import json
from uuid import uuid4

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from sensorlab.identity import TrustedRegistry, key_identifier, provision, raw_public
from sensorlab.playback import PlaybackEngine
from sensorlab.signing import (
    DOMAIN,
    SignedTelemetrySink,
    SigningService,
    parse_unsigned,
    signing_bytes,
    verify_envelope,
    verify_json,
)
from sensorlab.transport import NDJSONSink
from sensorlab.wire import SecurityError, decode64, encode64

PASSWORD = b"signing-synthetic-password-only"


@pytest.fixture(scope="module")
def signing_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("signing-tests") / "identity"
    provision(root, PASSWORD)
    return root, SigningService.unlock(root, PASSWORD)


@pytest.fixture
def telemetry(dataset, clock, sink):
    PlaybackEngine(dataset, sink, speed=0, clock=clock).run()
    return sink.messages[0]


def test_sign_verify_and_direct_cryptography_verification(signing_store, telemetry):
    _, service = signing_store
    envelope = service.sign(telemetry)
    assert verify_envelope(envelope, service.registry) == telemetry
    entry = service.registry.active_entry(telemetry.sensor_id)
    Ed25519PublicKey.from_public_bytes(decode64(entry.public_key, 32)).verify(
        decode64(envelope.signature, 64), signing_bytes(telemetry, entry.key_id)
    )
    # Ed25519 is deterministic for identical key and serialized bytes.
    assert service.sign(telemetry) == envelope
    assert verify_json(envelope.model_dump_json(), service.registry) == telemetry
    # Stateless verification intentionally permits replay; authorization is a future service.
    assert verify_json(envelope.model_dump_json(), service.registry) == telemetry


@pytest.mark.parametrize(
    "field",
    [
        "SOIL_MOISTURE_5",
        "SOIL_MOISTURE_10",
        "P_CALC",
        "LATITUDE",
        "LONGITUDE",
        "T_HR_AVG",
        "T_MAX",
        "T_MIN",
        "SOLARAD",
        "SUR_TEMP",
        "RH_HR_AVG",
        "SOIL_MOISTURE_20",
        "SOIL_MOISTURE_50",
        "SOIL_MOISTURE_100",
        "SOIL_TEMP_5",
        "SOIL_TEMP_10",
        "SOIL_TEMP_20",
        "SOIL_TEMP_50",
        "SOIL_TEMP_100",
    ],
)
def test_every_measurement_authenticated(signing_store, telemetry, field):
    _, service = signing_store
    data = service.sign(telemetry).model_dump(mode="json")
    value = data["telemetry"]["measurements"][field]
    data["telemetry"]["measurements"][field] = 0.123 if value is None else value + 0.01
    with pytest.raises(SecurityError, match="signature"):
        verify_json(json.dumps(data), service.registry)


@pytest.mark.parametrize(
    "field",
    [
        "sensor_id",
        "station",
        "session_id",
        "sequence",
        "observation_time",
        "sent_at",
        "schema_version",
        "key_id",
        "envelope_version",
        "algorithm",
        "serialization",
        "signature",
    ],
)
def test_identity_timestamps_and_envelope_authenticated(signing_store, telemetry, field):
    _, service = signing_store
    data = service.sign(telemetry).model_dump(mode="json")
    if field in {"key_id", "envelope_version", "algorithm", "serialization", "signature"}:
        data[field] = {
            "key_id": service.registry.sensors[1].key_id,
            "envelope_version": "2.0",
            "algorithm": "none",
            "serialization": "other",
            "signature": encode64(b"\x00" * 64),
        }[field]
    else:
        message = data["telemetry"]
        message[field] = {
            "sensor_id": "unknown",
            "station": "foreign",
            "session_id": str(uuid4()),
            "sequence": 2,
            "observation_time": "2025-01-01T02:00:00Z",
            "sent_at": "2026-10-23T00:00:01Z",
            "schema_version": "2.0",
        }[field]
    with pytest.raises(SecurityError):
        verify_json(json.dumps(data), service.registry)


def test_cannot_relabel_both_sensor_and_station(signing_store, telemetry):
    _, service = signing_store
    data = service.sign(telemetry).model_dump(mode="json")
    other = service.registry.sensors[1]
    data["telemetry"].update(sensor_id=other.sensor_id, station=other.station)
    data["key_id"] = other.key_id
    with pytest.raises(SecurityError, match="signature"):
        verify_json(json.dumps(data), service.registry)


def test_wrong_key_and_revoked_registry(signing_store, telemetry):
    _, service = signing_store
    envelope = service.sign(telemetry)
    other = Ed25519PrivateKey.generate()
    entry = service.registry.sensors[0].model_copy(
        update={
            "key_id": key_identifier(raw_public(other)),
            "public_key": encode64(raw_public(other)),
        }
    )
    wrong_registry = service.registry.model_copy(
        update={"sensors": (entry, *service.registry.sensors[1:])}
    )
    with pytest.raises(SecurityError, match="untrusted"):
        verify_envelope(envelope, wrong_registry)
    revoked_entry = service.registry.sensors[0].model_copy(update={"status": "revoked"})
    revoked = service.registry.model_copy(
        update={"sensors": (revoked_entry, *service.registry.sensors[1:])}
    )
    with pytest.raises(SecurityError, match="revoked"):
        verify_envelope(envelope, revoked)
    with pytest.raises(SecurityError, match="signature"):
        verify_envelope(
            envelope.model_copy(
                update={
                    "signature": encode64(other.sign(signing_bytes(telemetry, envelope.key_id)))
                }
            ),
            service.registry,
        )


def test_canonical_bytes_and_equivalent_json(signing_store, telemetry):
    _, service = signing_store
    envelope = service.sign(telemetry)
    body = signing_bytes(telemetry, envelope.key_id)
    assert body.startswith(DOMAIN)
    canonical = json.loads(body[len(DOMAIN) :])
    value = telemetry.measurements.P_CALC
    assert canonical["telemetry"]["measurements"]["P_CALC"] == (
        None if value is None else float(value).hex()
    )
    assert canonical["telemetry"]["sent_at"].endswith(".000000Z")
    assert b" " not in body
    reordered = dict(reversed(list(envelope.model_dump(mode="json").items())))
    reordered["telemetry"]["sent_at"] = "2026-10-23T00:00:00.000000+00:00"
    assert verify_json(json.dumps(reordered, indent=4), service.registry) == telemetry


def test_negative_zero_and_null_are_distinct(signing_store, telemetry):
    _, service = signing_store
    key_id = service.registry.sensors[0].key_id
    messages = []
    for value in (0.0, -0.0, None):
        data = telemetry.model_dump(mode="json")
        data["measurements"]["P_CALC"] = value
        messages.append(parse_unsigned(json.dumps(data)))
    assert len({signing_bytes(message, key_id) for message in messages}) == 3


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate",
        "nested_duplicate",
        "nan",
        "unsigned",
        "extra_public_key",
        "short_signature",
        "blank",
        "list",
    ],
)
def test_ambiguous_malformed_and_untrusted_envelopes(signing_store, telemetry, problem):
    _, service = signing_store
    data = service.sign(telemetry).model_dump(mode="json")
    if problem == "duplicate":
        text = '{"algorithm":"Ed25519",' + json.dumps(data)[1:]
    elif problem == "nested_duplicate":
        text = json.dumps(data).replace('"sequence": 1', '"sequence": 1, "sequence": 1')
    elif problem == "nan":
        data["telemetry"]["measurements"]["P_CALC"] = float("nan")
        text = json.dumps(data)
    elif problem == "unsigned":
        text = telemetry.model_dump_json()
    elif problem == "extra_public_key":
        data["public_key"] = service.registry.sensors[0].public_key
        text = json.dumps(data)
    elif problem == "short_signature":
        data["signature"] = encode64(b"short")
        text = json.dumps(data)
    else:
        text = "" if problem == "blank" else "[]"
    with pytest.raises(SecurityError):
        verify_json(text, service.registry)


def test_signing_adapter_preserves_playback_and_nulls(dataset, clock, signing_store):
    _, service = signing_store
    stream = io.StringIO()
    status = PlaybackEngine(
        dataset, SignedTelemetrySink(service, NDJSONSink(stream)), speed=3600, clock=clock
    ).run()
    assert status["transmitted"] == 18
    messages = [verify_json(line, service.registry) for line in stream.getvalue().splitlines()]
    assert sum(m.measurements.P_CALC is None for m in messages) == 1
    for message, observation in zip(messages, dataset.observations(), strict=True):
        assert message.measurements == observation.measurements
        assert message.observation_time == observation.time


def test_rfc8032_ed25519_known_answer():
    # RFC 8032 section 7.1, TEST 1: public key and signature only, no private seed.
    key = Ed25519PublicKey.from_public_bytes(
        bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    )
    expected = bytes.fromhex(
        "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e065224901555f"
        "b8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
    )
    key.verify(expected, b"")
    with pytest.raises(InvalidSignature):
        key.verify(expected, b"modified")


@pytest.mark.real_dataset
def test_real_24_hour_signed_replay_and_tampering(real_release, clock, signing_store):
    import hashlib

    from sensorlab.dataset import DetectorDataset, parse_utc

    _, service = signing_store
    inputs = [real_release / name for name in ("detector_input.parquet", "data_contract.json")]
    before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    selected = DetectorDataset.load(real_release).select(
        start=parse_utc("2025-01-01T01:00:00Z"), end=parse_utc("2025-01-02T01:00:00Z")
    )
    stream = io.StringIO()
    status = PlaybackEngine(
        selected, SignedTelemetrySink(service, NDJSONSink(stream)), speed=3600, clock=clock
    ).run()
    assert status["transmitted"] == 144
    public_registry = TrustedRegistry.model_validate_json(service.registry.model_dump_json())
    messages = [verify_json(line, public_registry) for line in stream.getvalue().splitlines()]
    for message, observation in zip(messages, selected.observations(), strict=True):
        assert message.measurements == observation.measurements
        assert message.observation_time == observation.time
    for sensor in status["sensors"]:
        assert sensor["transmitted"] == 24
        assert [m.sequence for m in messages if m.sensor_id == sensor["sensor_id"]] == list(
            range(1, 25)
        )
    altered = json.loads(stream.getvalue().splitlines()[0])
    altered["telemetry"]["measurements"]["SOIL_MOISTURE_5"] += 0.001
    with pytest.raises(SecurityError, match="invalid Ed25519 signature"):
        verify_json(json.dumps(altered), public_registry)
    assert {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs} == before


def test_committed_public_interoperability_vector():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "validation"
    vector = json.loads((root / "signed-telemetry-v1-vector.json").read_text())
    results = json.loads((root / "sensor_identity_results.json").read_text())
    trusted = TrustedRegistry.model_validate(results["registry"])
    message = verify_json(json.dumps(vector["envelope"]), trusted)
    expected_bytes = decode64(
        vector["signing_bytes_base64"], len(signing_bytes(message, vector["envelope"]["key_id"]))
    )
    assert expected_bytes == signing_bytes(message, vector["envelope"]["key_id"])
    # Verify the published exact bytes directly, without the SensorLab verifier or signer.
    Ed25519PublicKey.from_public_bytes(decode64(vector["public_key"], 32)).verify(
        decode64(vector["envelope"]["signature"], 64), expected_bytes
    )


def test_wrong_signer_fails_before_output(dataset, clock, signing_store):
    from sensorlab.signing import Ed25519Signer

    _, service = signing_store
    wrong = SigningService(
        service.registry,
        {service.registry.sensors[0].sensor_id: Ed25519Signer(Ed25519PrivateKey.generate())},
    )
    stream = io.StringIO()
    engine = PlaybackEngine(
        dataset, SignedTelemetrySink(wrong, NDJSONSink(stream)), speed=0, clock=clock
    )
    with pytest.raises(SecurityError, match="invalid Ed25519 signature"):
        engine.run()
    assert engine.status()["state"] == "failed"
    assert engine.status()["transmitted"] == 0
    assert stream.getvalue() == ""
