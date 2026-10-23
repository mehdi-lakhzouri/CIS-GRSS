import os

import pytest
from cryptography.hazmat.primitives import serialization
from pydantic import ValidationError

from sensorlab.identity import (
    TrustedRegistry,
    key_identifier,
    load_private_key,
    provision,
    raw_public,
    revoke,
)
from sensorlab.registry import SENSOR_STATIONS
from sensorlab.wire import SecurityError, decode64, encode64

PASSWORD = b"synthetic-test-password-not-production"


@pytest.fixture(scope="module")
def identity_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("identity-tests") / "new-store"
    registry = provision(root, PASSWORD)
    return root, registry


def test_six_secure_independent_encrypted_keys(identity_store):
    root, registry = identity_store
    assert len(registry.sensors) == len({entry.public_key for entry in registry.sensors}) == 6
    assert TrustedRegistry.load(root / "registry.json") == registry
    assert (root / ".gitignore").read_text() == "*\n"
    assert {entry.sensor_id: entry.station for entry in registry.sensors} == dict(SENSOR_STATIONS)
    for entry in registry.sensors:
        path = root / "private" / f"{entry.key_id}.private.pem"
        encrypted = path.read_bytes()
        assert encrypted.startswith(b"-----BEGIN ENCRYPTED PRIVATE KEY-----")
        with pytest.raises(TypeError):
            serialization.load_pem_private_key(encrypted, password=None)
        key = load_private_key(root, entry, PASSWORD)
        assert raw_public(key) == decode64(entry.public_key, 32)
        assert entry.key_id == key_identifier(raw_public(key))
        if os.name != "nt":
            assert path.stat().st_mode & 0o777 == 0o600
            assert root.stat().st_mode & 0o777 == 0o700
    assert "PRIVATE KEY" not in (root / "registry.json").read_text()


def test_no_overwrite_and_no_password_leakage(identity_store):
    root, registry = identity_store
    snapshot = (root / "registry.json").read_bytes()
    with pytest.raises(FileExistsError):
        provision(root, PASSWORD)
    assert (root / "registry.json").read_bytes() == snapshot
    with pytest.raises(SecurityError) as error:
        load_private_key(root, registry.sensors[0], b"wrong-secret-password")
    assert "wrong-secret-password" not in str(error.value)


def test_short_password_creates_no_keys(tmp_path):
    root = tmp_path / "never-created"
    with pytest.raises(SecurityError):
        provision(root, b"short")
    assert not root.exists()


@pytest.mark.parametrize(
    "problem",
    [
        "mapping",
        "duplicate_keys",
        "missing_sensor",
        "fingerprint",
        "public_key",
        "status",
        "extra_private_field",
    ],
)
def test_invalid_trusted_registry(identity_store, problem):
    _, registry = identity_store
    data = registry.model_dump(mode="json")
    entry = data["sensors"][0]
    if problem == "mapping":
        entry["station"] = "other"
    elif problem == "duplicate_keys":
        entry["key_id"] = data["sensors"][1]["key_id"]
        entry["public_key"] = data["sensors"][1]["public_key"]
    elif problem == "missing_sensor":
        data["sensors"].pop()
    elif problem == "fingerprint":
        entry["key_id"] = "ed25519-" + "0" * 64
    elif problem == "public_key":
        entry["public_key"] = encode64(b"short")
    elif problem == "status":
        entry["status"] = "disabled"
    else:
        entry["private_key"] = "must-not-be-here"
    with pytest.raises((ValidationError, SecurityError)):
        TrustedRegistry.model_validate(data)


def test_private_key_must_match_registry(tmp_path, identity_store):
    root, registry = identity_store
    replacement = tmp_path / "wrong-store"
    replacement.mkdir(mode=0o700)
    (replacement / "private").mkdir(mode=0o700)
    entry = registry.sensors[0]
    wrong = (root / "private" / f"{registry.sensors[1].key_id}.private.pem").read_bytes()
    path = replacement / "private" / f"{entry.key_id}.private.pem"
    path.write_bytes(wrong)
    path.chmod(0o600)
    with pytest.raises(SecurityError):
        load_private_key(replacement, entry, PASSWORD)


def test_revocation_and_public_only_verification_registry(tmp_path, identity_store):
    _, registry = identity_store
    path = tmp_path / "public-registry.json"
    path.write_text(registry.model_dump_json())
    revoked = revoke(path, registry.sensors[0].sensor_id)
    assert revoked.sensors[0].status == "revoked"
    assert TrustedRegistry.load(path) == revoked
    with pytest.raises(SecurityError, match="revoked"):
        revoked.active_entry(registry.sensors[0].sensor_id)
    with pytest.raises(SecurityError, match="unknown"):
        revoked.active_entry("unknown")
    with pytest.raises(SecurityError):
        revoke(path, "unknown")


def test_duplicate_registry_json_rejected(tmp_path, identity_store):
    _, registry = identity_store
    path = tmp_path / "ambiguous.json"
    data = registry.model_dump_json()
    path.write_text('{"algorithm":"Ed25519",' + data[1:])
    with pytest.raises(SecurityError):
        TrustedRegistry.load(path)


def test_independent_random_provisionings(tmp_path, identity_store):
    _, original = identity_store
    fresh = provision(tmp_path / "independent", PASSWORD)
    assert not {entry.key_id for entry in fresh.sensors} & {
        entry.key_id for entry in original.sensors
    }
