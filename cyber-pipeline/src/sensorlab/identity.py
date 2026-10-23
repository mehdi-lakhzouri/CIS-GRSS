"""Local provisioning, encrypted private storage, and an operator-trusted public registry."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sensorlab.registry import SENSOR_STATIONS
from sensorlab.wire import SecurityError, decode64, encode64, strict_json

DEFAULT_IDENTITY_DIR = Path(".sensorlab/identity")
KEY_PATTERN = r"^ed25519-[a-f0-9]{64}$"


def key_identifier(public_bytes: bytes) -> str:
    return "ed25519-" + hashlib.sha256(public_bytes).hexdigest()


def raw_public(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    sensor_id: str
    station: str
    key_id: str = Field(pattern=KEY_PATTERN)
    public_key: str
    status: Literal["active", "revoked"]

    @model_validator(mode="after")
    def valid_identity(self) -> RegistryEntry:
        if SENSOR_STATIONS.get(self.sensor_id) != self.station:
            raise ValueError("registry sensor/station mismatch")
        if key_identifier(decode64(self.public_key, 32)) != self.key_id:
            raise ValueError("key identifier does not match the public key fingerprint")
        return self


class TrustedRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    registry_version: Literal["1.0"]
    algorithm: Literal["Ed25519"]
    sensors: tuple[RegistryEntry, ...]

    @model_validator(mode="after")
    def six_independent_keys(self) -> TrustedRegistry:
        if len(self.sensors) != 6 or {entry.sensor_id for entry in self.sensors} != set(
            SENSOR_STATIONS
        ):
            raise ValueError("registry must contain exactly the six known sensors")
        if len({entry.key_id for entry in self.sensors}) != 6:
            raise ValueError("each sensor requires an independent key")
        return self

    @classmethod
    def load(cls, path: Path) -> TrustedRegistry:
        try:
            return cls.model_validate(strict_json(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise SecurityError("cannot load a valid trusted public registry") from exc

    def active_entry(self, sensor_id: str, key_id: str | None = None) -> RegistryEntry:
        entry = next((entry for entry in self.sensors if entry.sensor_id == sensor_id), None)
        if entry is None:
            raise SecurityError("unknown sensor")
        if key_id is not None and entry.key_id != key_id:
            raise SecurityError("untrusted key identifier for sensor")
        if entry.status != "active":
            raise SecurityError("sensor key is revoked")
        return entry


def protect_directory(path: Path) -> None:
    """Restrict a newly created directory before writing any key material."""
    if os.name == "nt":
        try:
            identity = subprocess.run(
                ["whoami", "/user", "/fo", "csv", "/nh"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            match = re.search(r"S-1-5-(?:\d+-)*\d+", identity)
            if match is None:
                raise SecurityError("cannot determine current Windows user SID")
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"*{match.group()}:(OI)(CI)F"],
                capture_output=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SecurityError("cannot restrict Windows identity directory ACL") from exc
    else:
        path.chmod(0o700)


def _check_private_location(path: Path, directory: bool = False) -> None:
    if path.is_symlink() or path.is_junction():
        raise SecurityError("linked private key storage is not allowed")
    info = path.stat()
    if directory and not stat.S_ISDIR(info.st_mode):
        raise SecurityError("invalid private key directory")
    if not directory and not stat.S_ISREG(info.st_mode):
        raise SecurityError("invalid private key file")
    if os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o077):
        raise SecurityError("private storage must be owned by the user with no group/other access")


def provision(directory: Path, password: bytes) -> TrustedRegistry:
    """Create a fresh six-key store. Never reuse, replace, or print existing secrets."""
    if len(password) < 16:
        raise SecurityError("key password must contain at least 16 UTF-8 bytes")
    # Exclusive root creation prevents accidental overwrite and avoids modifying old ACLs.
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    protect_directory(directory)
    (directory / ".gitignore").write_text("*\n", encoding="utf-8")
    private_dir = directory / "private"
    private_dir.mkdir(mode=0o700)
    entries = []
    for sensor_id, station in SENSOR_STATIONS.items():
        private_key = Ed25519PrivateKey.generate()
        public_bytes = raw_public(private_key)
        key_id = key_identifier(public_bytes)
        encrypted = private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.BestAvailableEncryption(password),
        )
        path = private_dir / f"{key_id}.private.pem"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encrypted)
            stream.flush()
            os.fsync(stream.fileno())
        entries.append(
            RegistryEntry(
                sensor_id=sensor_id,
                station=station,
                key_id=key_id,
                public_key=encode64(public_bytes),
                status="active",
            )
        )
    registry = TrustedRegistry(registry_version="1.0", algorithm="Ed25519", sensors=tuple(entries))
    with (directory / "registry.json").open("x", encoding="utf-8") as stream:
        stream.write(registry.model_dump_json(indent=2) + "\n")
    return registry


def load_private_key(directory: Path, entry: RegistryEntry, password: bytes) -> Ed25519PrivateKey:
    path = directory / "private" / f"{entry.key_id}.private.pem"
    try:
        for folder in (directory, directory / "private"):
            _check_private_location(folder, directory=True)
        _check_private_location(path)
        key = serialization.load_pem_private_key(path.read_bytes(), password=password)
        if not isinstance(key, Ed25519PrivateKey) or raw_public(key) != decode64(
            entry.public_key, 32
        ):
            raise SecurityError("private key does not match the trusted registry")
        return key
    except (OSError, ValueError, TypeError) as exc:
        raise SecurityError("cannot unlock a matching Ed25519 private key") from exc


def revoke(registry_path: Path, sensor_id: str) -> TrustedRegistry:
    registry = TrustedRegistry.load(registry_path)
    if sensor_id not in SENSOR_STATIONS:
        raise SecurityError("unknown sensor")
    updated = registry.model_copy(
        update={
            "sensors": tuple(
                entry.model_copy(update={"status": "revoked"})
                if entry.sensor_id == sensor_id
                else entry
                for entry in registry.sensors
            )
        }
    )
    descriptor, temporary = tempfile.mkstemp(prefix=".registry-", dir=registry_path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(updated.model_dump_json(indent=2) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(registry_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return updated
