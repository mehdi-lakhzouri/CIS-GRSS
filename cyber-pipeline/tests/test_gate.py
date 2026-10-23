import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from sensorlab.gate.api import create_app
from sensorlab.gate.repository import SQLiteGateRepository
from sensorlab.identity import revoke


def submit(client, service, message):
    return client.post("/v1/telemetry", json=service.sign(message).model_dump(mode="json"))


def test_valid_packet_and_structured_audit(gate_environment):
    _, client, repository, message, signer = gate_environment
    response = submit(client, signer, message)
    assert response.status_code == 202
    decision = response.json()
    assert decision["decision"] == "ACCEPT" and decision["code"] == "OK"
    assert decision["warnings"] == []
    assert len(decision["packet_digest"]) == 64
    assert decision["station"] == message.station
    event = repository.events()[0]
    assert event["decision"] == "ACCEPT" and event["sequence"] == 1
    assert event["session_id"] == str(message.session_id)
    assert "measurements" not in json.dumps(event)
    assert "signature" not in json.dumps(event)
    assert repository.status()["outbox_count"] == 1
    assert repository.accepted_after(0)[0]["telemetry"] == message.model_dump(mode="json")


@pytest.mark.parametrize(
    "problem,code",
    [
        ("measurement", "E_BAD_SIG"),
        ("unknown_sensor", "E_UNKNOWN_KEY"),
        ("station", "E_UNKNOWN_KEY"),
        ("wrong_key", "E_UNKNOWN_KEY"),
        ("unsigned", "E_SCHEMA"),
        ("missing_sequence", "E_SCHEMA"),
        ("extra_field", "E_SCHEMA"),
        ("huge_sequence", "E_SCHEMA"),
    ],
)
def test_ordered_gate_rejections(gate_environment, problem, code):
    _, client, repository, message, signer = gate_environment
    packet = signer.sign(message).model_dump(mode="json")
    if problem == "measurement":
        packet["telemetry"]["measurements"]["SOIL_MOISTURE_5"] += 0.001
    elif problem == "unknown_sensor":
        packet["telemetry"]["sensor_id"] = "unregistered-sensor"
    elif problem == "station":
        packet["telemetry"]["station"] = "other-station"
    elif problem == "wrong_key":
        packet["key_id"] = signer.registry.sensors[1].key_id
    elif problem == "unsigned":
        packet = message.model_dump(mode="json")
    elif problem == "missing_sequence":
        del packet["telemetry"]["sequence"]
    elif problem == "extra_field":
        packet["public_key"] = signer.registry.sensors[0].public_key
    else:
        packet["telemetry"]["sequence"] = 2**63
    result = client.post("/v1/telemetry", json=packet)
    assert result.json()["code"] == code
    assert result.json()["decision"] == "REJECT"
    assert repository.status()["accepted"] == 0
    assert repository.status()["rejected"] == 1


def test_key_revocation_is_reloaded(gate_environment):
    config, client, _, message, signer = gate_environment
    revoke(config.registry, message.sensor_id)
    assert submit(client, signer, message).json()["code"] == "E_REVOKED_KEY"


def test_missing_registry_fails_closed(gate_environment):
    config, client, repository, message, signer = gate_environment
    config.registry.write_text("corrupted")
    assert submit(client, signer, message).status_code == 503
    assert repository.events()[0]["code"] == "E_REGISTRY"


def test_duplicate_gap_and_out_of_order(gate_environment):
    _, client, repository, message, signer = gate_environment
    assert submit(client, signer, message).status_code == 202
    assert submit(client, signer, message).json()["code"] == "E_REPLAY_SEQ"
    gap = submit(client, signer, message.model_copy(update={"sequence": 4}))
    assert gap.status_code == 202
    assert gap.json()["warnings"] == ["W_SEQ_GAP"]
    assert gap.json()["sequence_gap"] == 2
    assert (
        submit(client, signer, message.model_copy(update={"sequence": 3})).json()["code"]
        == "E_REPLAY_SEQ"
    )
    assert (
        submit(client, signer, message.model_copy(update={"sequence": 5})).json()["warnings"] == []
    )
    assert repository.status()["accepted"] == 3
    assert repository.status()["sequence_gap_events"] == 1


def test_concurrent_duplicates_only_one_accepted(gate_environment):
    _, client, repository, message, signer = gate_environment
    packet = signer.sign(message).model_dump(mode="json")
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: client.post("/v1/telemetry", json=packet), range(12)))
    assert sum(response.status_code == 202 for response in responses) == 1
    assert sum(response.status_code == 409 for response in responses) == 11
    assert repository.status()["accepted"] == repository.status()["outbox_count"] == 1
    assert repository.status()["rejected"] == 11


def test_restart_keeps_state_and_cannot_reauthorize(gate_environment, clock):
    config, client, repository, message, signer = gate_environment
    assert submit(client, signer, message).status_code == 202
    restarted = TestClient(create_app(config, clock), base_url="https://testserver")
    assert submit(restarted, signer, message).json()["code"] == "E_REPLAY_SEQ"
    with pytest.raises(ValueError, match="never resets"):
        repository.authorize(
            message.session_id,
            list(signer.registry.sensors),
            clock.now(),
            clock.now(),
            clock.now() + timedelta(hours=1),
        )
    assert repository.status()["sensors"][0]["last_sequence"] == 1


@pytest.mark.parametrize(
    "case", ["unknown", "expired", "revoked", "not_yet_active", "sensor_scope"]
)
def test_session_authorization(gate_environment, clock, case):
    _, client, repository, message, signer = gate_environment
    if case == "unknown":
        message = message.model_copy(update={"session_id": uuid4()})
    elif case == "expired":
        clock.elapsed += 3601
    elif case == "revoked":
        repository.revoke_session(message.session_id)
    else:
        new_id = uuid4()
        entries = (
            [signer.registry.sensors[1]]
            if case == "sensor_scope"
            else list(signer.registry.sensors)
        )
        repository.authorize(
            new_id,
            entries,
            clock.now(),
            clock.now() + timedelta(minutes=5),
            clock.now() + timedelta(hours=1),
        )
        message = message.model_copy(update={"session_id": new_id})
    response = submit(client, signer, message)
    assert response.json()["code"] == "E_UNAUTHORIZED_SESSION"
    assert repository.status()["accepted"] == 0


@pytest.mark.parametrize("offset,accept", [(-121, False), (6, False), (-120, True), (5, True)])
def test_signed_sent_at_freshness_not_observation_time(gate_environment, clock, offset, accept):
    _, client, repository, message, signer = gate_environment
    message = message.model_copy(update={"sent_at": clock.now() + timedelta(seconds=offset)})
    response = submit(client, signer, message)
    assert response.json()["code"] == ("OK" if accept else "E_STALE")
    assert message.observation_time.year == 2025  # Historic data accepted when sent_at is fresh.
    assert repository.status()["accepted"] == int(accept)


def test_stale_packet_does_not_consume_sequence(gate_environment, clock):
    _, client, _, message, signer = gate_environment
    stale = message.model_copy(update={"sent_at": clock.now() - timedelta(hours=1)})
    assert submit(client, signer, stale).json()["code"] == "E_STALE"
    assert submit(client, signer, message).status_code == 202


def test_schema_json_and_payload_limits(gate_environment):
    config, client, repository, message, signer = gate_environment
    for body in (b"not json", b"\xff", b"{}", b'{"algorithm":1,"algorithm":1}', b"[]"):
        assert (
            client.post(
                "/v1/telemetry", content=body, headers={"content-type": "application/json"}
            ).json()["code"]
            == "E_SCHEMA"
        )
    response = client.post(
        "/v1/telemetry",
        content=b"a" * (config.max_payload_bytes + 100),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["digest_scope"] == "received-prefix"
    assert (
        client.post(
            "/v1/telemetry", content="{}", headers={"content-type": "text/plain"}
        ).status_code
        == 415
    )
    response = client.post(
        "/v1/telemetry",
        json=signer.sign(message).model_dump(mode="json"),
        headers={"content-encoding": "gzip"},
    )
    assert response.status_code == 415
    assert repository.status()["accepted"] == 0


def test_https_cannot_be_spoofed_with_forwarded_header(gate_environment, clock):
    config, _, _, message, signer = gate_environment
    plain = TestClient(create_app(config, clock), base_url="http://testserver")
    result = plain.post(
        "/v1/telemetry",
        json=signer.sign(message).model_dump(mode="json"),
        headers={"x-forwarded-proto": "https"},
    )
    assert result.json()["code"] == "E_HTTPS_REQUIRED"


def test_outbox_failure_rolls_back_audit_and_sequence(gate_environment):
    _, client, repository, message, signer = gate_environment
    with repository.connection() as connection:
        connection.executescript("""
            CREATE TRIGGER fail_outbox BEFORE INSERT ON accepted_observations
            BEGIN SELECT RAISE(ABORT,'test failure'); END;
        """)
    assert submit(client, signer, message).status_code == 503
    assert repository.status()["accepted"] == repository.status()["outbox_count"] == 0
    assert repository.status()["sensors"][0]["last_sequence"] == 0
    with repository.connection() as connection:
        connection.execute("DROP TRIGGER fail_outbox")
    assert submit(client, signer, message).status_code == 202


def test_unrelated_database_not_overwritten(tmp_path):
    path = tmp_path / "other.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE existing_data(value TEXT)")
    with pytest.raises(ValueError, match="unrelated"):
        SQLiteGateRepository(path)
