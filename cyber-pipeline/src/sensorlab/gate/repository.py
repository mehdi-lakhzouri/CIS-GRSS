"""Persistent policy repository. One transaction for session/sequence/freshness/audit/outbox."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from sensorlab.clock import Clock
from sensorlab.gate.models import Decision, GateConfig, Packet, utc_text
from sensorlab.identity import RegistryEntry, protect_directory
from sensorlab.telemetry import TelemetryMessage


class GateRepository(Protocol):
    def reject(self, packet: Packet, code: str, reason: str, status: int) -> Decision: ...
    def admit(
        self,
        message: TelemetryMessage,
        key_id: str,
        packet: Packet,
        config: GateConfig,
        clock: Clock,
    ) -> Decision: ...


class Stage2Source(Protocol):
    """Future consumer interface over durable accepted observations, ordered by decision ID."""

    def accepted_after(self, decision_id: int, limit: int = 100) -> list[dict]: ...


class SQLiteGateRepository:
    def __init__(self, path: Path):
        self.path = path.resolve()
        if not self.path.parent.exists():
            self.path.parent.mkdir(mode=0o700, parents=True)
            protect_directory(self.path.parent)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(descriptor)
        with self.connection() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("unsupported Crypto Gate database version")
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            if version == 0 and tables:
                raise ValueError("refusing to initialize an unrelated database")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
                    not_before TEXT NOT NULL, expires_at TEXT NOT NULL,
                    revoked INTEGER NOT NULL DEFAULT 0 CHECK(revoked IN (0,1)));
                CREATE TABLE IF NOT EXISTS session_sensors (
                    session_id TEXT NOT NULL REFERENCES sessions(session_id),
                    sensor_id TEXT NOT NULL, key_id TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(session_id,sensor_id));
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL, sensor_id TEXT, station TEXT, session_id TEXT,
                    sequence INTEGER, decision TEXT NOT NULL, code TEXT NOT NULL,
                    reason TEXT NOT NULL, packet_digest TEXT NOT NULL,
                    digest_scope TEXT NOT NULL, warnings TEXT NOT NULL,
                    sequence_gap INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS accepted_observations (
                    decision_id INTEGER PRIMARY KEY REFERENCES decisions(decision_id),
                    telemetry_json TEXT NOT NULL);
                PRAGMA user_version=1;
            """)

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA trusted_schema=OFF")
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self):
        with self.connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _record(connection: sqlite3.Connection, result: Decision) -> Decision:
        cursor = connection.execute(
            """
            INSERT INTO decisions(timestamp,sensor_id,station,session_id,sequence,decision,
                code,reason,packet_digest,digest_scope,warnings,sequence_gap)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        """,
            (
                utc_text(result.timestamp),
                result.sensor_id,
                result.station,
                result.session_id,
                result.sequence,
                result.decision,
                result.code,
                result.reason,
                result.packet_digest,
                result.digest_scope,
                json.dumps(result.warnings),
                result.sequence_gap,
            ),
        )
        return result.model_copy(update={"decision_id": cursor.lastrowid})

    def reject(self, packet: Packet, code: str, reason: str, status: int) -> Decision:
        result = Decision(
            **packet.model_dump(), decision="REJECT", code=code, reason=reason, http_status=status
        )
        with self.transaction() as connection:
            return self._record(connection, result)

    def authorize(
        self,
        session_id: UUID,
        entries: list[RegistryEntry],
        now: datetime,
        not_before: datetime,
        expires_at: datetime,
    ) -> None:
        if expires_at <= now or expires_at <= not_before:
            raise ValueError("session expiration must be future and after not-before")
        if not entries or len({entry.sensor_id for entry in entries}) != len(entries):
            raise ValueError("authorize at least one distinct sensor")
        if any(entry.status != "active" for entry in entries):
            raise ValueError("only active keys may be authorized")
        with self.transaction() as connection:
            try:
                connection.execute(
                    "INSERT INTO sessions VALUES(?,?,?,?,0)",
                    (str(session_id), utc_text(now), utc_text(not_before), utc_text(expires_at)),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "session already exists; authorization never resets sequences"
                ) from exc
            connection.executemany(
                "INSERT INTO session_sensors VALUES(?,?,?,0)",
                [(str(session_id), entry.sensor_id, entry.key_id) for entry in entries],
            )

    def revoke_session(self, session_id: UUID) -> None:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE sessions SET revoked=1 WHERE session_id=?", (str(session_id),)
            )
            if cursor.rowcount != 1:
                raise ValueError("unknown session")

    def admit(
        self,
        message: TelemetryMessage,
        key_id: str,
        packet: Packet,
        config: GateConfig,
        clock: Clock,
    ) -> Decision:
        with self.transaction() as connection:
            # Time is sampled AFTER acquiring the write lock: expiry/freshness cannot go stale
            # while this request waits behind a concurrent transaction.
            now = clock.now()
            now_text = utc_text(now)
            session = connection.execute(
                """
                SELECT s.revoked,s.not_before,s.expires_at,ss.last_sequence,ss.key_id
                FROM sessions s JOIN session_sensors ss USING(session_id)
                WHERE s.session_id=? AND ss.sensor_id=?
            """,
                (str(message.session_id), message.sensor_id),
            ).fetchone()
            code, reason, status = "OK", "accepted", 202
            gap = 0
            if session is None or session["key_id"] != key_id:
                code, reason, status = (
                    "E_UNAUTHORIZED_SESSION",
                    "session/sensor/key not authorized",
                    403,
                )
            elif session["revoked"]:
                code, reason, status = "E_UNAUTHORIZED_SESSION", "session is revoked", 403
            elif not session["not_before"] <= now_text < session["expires_at"]:
                code, reason, status = (
                    "E_UNAUTHORIZED_SESSION",
                    "session is not active or expired",
                    403,
                )
            elif message.sequence <= session["last_sequence"]:
                code, reason, status = "E_REPLAY_SEQ", "sequence is duplicate or out of order", 409
            else:
                age = (now - message.sent_at).total_seconds()
                if age > config.max_age_seconds or age < -config.future_skew_seconds:
                    code, reason, status = "E_STALE", "sent_at is outside freshness tolerance", 422
                else:
                    gap = message.sequence - session["last_sequence"] - 1
            accepted = code == "OK"
            result = self._record(
                connection,
                Decision(
                    **packet.model_dump(),
                    decision="ACCEPT" if accepted else "REJECT",
                    code=code,
                    reason=reason,
                    http_status=status,
                    warnings=("W_SEQ_GAP",) if gap else (),
                    sequence_gap=gap,
                ),
            )
            if accepted:
                connection.execute(
                    """UPDATE session_sensors SET last_sequence=?
                    WHERE session_id=? AND sensor_id=?""",
                    (message.sequence, str(message.session_id), message.sensor_id),
                )
                connection.execute(
                    "INSERT INTO accepted_observations VALUES(?,?)",
                    (result.decision_id, message.model_dump_json()),
                )
            return result

    def status(self) -> dict:
        with self.connection() as connection:
            connection.execute(
                "BEGIN"
            )  # One read snapshot for consistent counters and sensor state.
            counts = dict(
                connection.execute(
                    "SELECT decision,COUNT(*) FROM decisions GROUP BY decision"
                ).fetchall()
            )
            codes = dict(connection.execute("SELECT code,COUNT(*) FROM decisions GROUP BY code"))
            sensors = [
                dict(row)
                for row in connection.execute("""
                SELECT ss.*,s.not_before,s.expires_at,s.revoked FROM session_sensors ss
                JOIN sessions s USING(session_id) ORDER BY session_id,sensor_id
            """)
            ]
            return {
                "accepted": counts.get("ACCEPT", 0),
                "rejected": counts.get("REJECT", 0),
                "codes": codes,
                "sequence_gap_events": connection.execute(
                    "SELECT COUNT(*) FROM decisions WHERE sequence_gap>0"
                ).fetchone()[0],
                "outbox_count": connection.execute(
                    "SELECT COUNT(*) FROM accepted_observations"
                ).fetchone()[0],
                "sensors": sensors,
            }

    def events(self, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 10000:
            raise ValueError("limit must be 1..10000")
        with self.connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM decisions ORDER BY decision_id DESC LIMIT ?", (limit,)
                )
            ]

    def accepted_after(self, decision_id: int, limit: int = 100) -> list[dict]:
        if not 1 <= limit <= 10000 or decision_id < 0:
            raise ValueError("invalid outbox cursor/limit")
        with self.connection() as connection:
            return [
                {"decision_id": row[0], "telemetry": json.loads(row[1])}
                for row in connection.execute(
                    """
                        SELECT decision_id,telemetry_json FROM accepted_observations
                        WHERE decision_id>? ORDER BY decision_id LIMIT ?
                    """,
                    (decision_id, limit),
                )
            ]
